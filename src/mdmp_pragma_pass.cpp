#include "mdmp_pragma_pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

PreservedAnalyses MDMPPragmaPass::run(Module &M, ModuleAnalysisManager &MAM) {
    bool changed = false;
    
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    for (auto &F : M) {
        if (!F.isDeclaration()) {
            AAResults &AA = FAM.getResult<AAManager>(F);
            
            changed |= runOnFunction(F, AA);
        }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MDMPPragmaPass::runOnFunction(Function &F, AAResults &AA) {
    PendingRequests.clear(); // Reset for each function
    transformPragmasToCalls(F, AA);
    return true; 
}

void MDMPPragmaPass::transformPragmasToCalls(Function &F, AAResults &AA) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    // Define all Runtime Functions
    FunctionCallee runtime_begin = M->getOrInsertFunction("mdmp_commregion_begin", Type::getVoidTy(Ctx));
    FunctionCallee runtime_end   = M->getOrInsertFunction("mdmp_commregion_end", Type::getVoidTy(Ctx));
    FunctionCallee runtime_sync  = M->getOrInsertFunction("mdmp_sync", Type::getVoidTy(Ctx));
    FunctionCallee runtime_init  = M->getOrInsertFunction("mdmp_init", Type::getVoidTy(Ctx));
    FunctionCallee runtime_final = M->getOrInsertFunction("mdmp_final", Type::getVoidTy(Ctx));
    FunctionCallee runtime_get_rank = M->getOrInsertFunction("mdmp_get_rank", Type::getInt32Ty(Ctx));
    FunctionCallee runtime_get_size = M->getOrInsertFunction("mdmp_get_size", Type::getInt32Ty(Ctx));
    // Send uses: Buffer, Count, Type, Actor, Peer, Tag
    FunctionCallee runtime_send = M->getOrInsertFunction("mdmp_send", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
        
    if (Function *FSend = dyn_cast<Function>(runtime_send.getCallee())) {
        FSend->addFnAttr(Attribute::NoUnwind);
        FSend->setMemoryEffects(MemoryEffects::readOnly());
        FSend->addParamAttr(0, Attribute::ReadOnly);
    }
    // Recv uses: Buffer, Count, Type, Actor, Peer, Tag
    FunctionCallee runtime_recv = M->getOrInsertFunction("mdmp_recv", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    if (Function *FRecv = dyn_cast<Function>(runtime_recv.getCallee())) {
        FRecv->addFnAttr(Attribute::NoUnwind);
    }

    std::vector<Instruction*> toDelete;

    for (auto &BB : F) {
        for (auto &I : BB) {
            auto *CI = dyn_cast<CallInst>(&I);
            if (!CI || !CI->getCalledFunction()) continue;
            
            StringRef Name = CI->getCalledFunction()->getName();
            IRBuilder<> Builder(CI);

            if (Name == "__mdmp_marker_send" || Name == "__mdmp_marker_recv") {
                Value *BufferPtr = CI->getArgOperand(0);
                Value *CountVal  = CI->getArgOperand(1);
                Value *TypeVal   = CI->getArgOperand(2);
                Value *BytesVal  = CI->getArgOperand(3); // Stripped out for backend
                Value *ActorRank = CI->getArgOperand(4);
                Value *PeerRank  = CI->getArgOperand(5);
                Value *TagVal    = CI->getArgOperand(6); 

                LocationSize LocSize = LocationSize::beforeOrAfterPointer();
                if (auto *ConstBytes = dyn_cast<ConstantInt>(BytesVal)) {
                    LocSize = LocationSize::precise(ConstBytes->getZExtValue());
                }
                MemoryLocation Loc(BufferPtr, LocSize);

                bool isSend = (Name == "__mdmp_marker_send");
                
                // Pass the TagVal directly to the backend runtime call
                CallInst *NewCall = Builder.CreateCall(
                    isSend ? runtime_send : runtime_recv, 
                    {BufferPtr, CountVal, TypeVal, ActorRank, PeerRank, TagVal}
                );
                
                CI->replaceAllUsesWith(NewCall);
                hoistInitiation(NewCall, Loc, AA, isSend);
                PendingRequests.push_back({Loc, NewCall});
                toDelete.push_back(CI);
            } 
            else if (Name == "__mdmp_marker_commregion_begin") {
                CallInst *NewCall = Builder.CreateCall(runtime_begin);
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_commregion_end") {
                CallInst *NewEnd = Builder.CreateCall(runtime_end);
                injectWaitsForRegion(NewEnd, AA, Ctx, M);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_get_rank") {
                CallInst *NewCall = Builder.CreateCall(runtime_get_rank);
                CI->replaceAllUsesWith(NewCall); 
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_get_size") {
                CallInst *NewCall = Builder.CreateCall(runtime_get_size);
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_init") {
                Builder.CreateCall(runtime_init);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_final") {
                Builder.CreateCall(runtime_final);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_sync") {
                Builder.CreateCall(runtime_sync);
                toDelete.push_back(CI);
            }
        }
    }
    for (Instruction *I : toDelete) I->eraseFromParent();
}

// Move communications to the earliest possible times (helps that they are nonblocking)
bool MDMPPragmaPass::hoistInitiation(CallInst *CommCall, MemoryLocation Loc, AAResults &AA, bool isSend) {
    // Establish the absolute ceiling (the latest instruction that generates an argument)
    Instruction *OpBound = nullptr;
    for (Use &U : CommCall->operands()) {
        if (auto *OpI = dyn_cast<Instruction>(U.get())) {
            if (OpI->getParent() == CommCall->getParent()) {
                if (!OpBound || OpBound->comesBefore(OpI)) {                   
                    OpBound = OpI;
                }
            }
        }
    }

    Instruction *InsertPoint = CommCall;
    BasicBlock::iterator it(CommCall);
    
    while (it != CommCall->getParent()->begin()) {
        Instruction *Prev = &*(--it);

        // Do not hoist past the instructions that generate our arguments
        if (OpBound && (Prev == OpBound || Prev->comesBefore(OpBound))) {
            break;
        }

        // Do not hoist past the start of the region
        if (auto *PrevCall = dyn_cast<CallInst>(Prev)) {
            if (Function *F = PrevCall->getCalledFunction()) {
                if (F->getName() == "__mdmp_marker_commregion_begin") {
                    break;
                }
            }
        }

        // Check for Memory Dependencies
        auto ModRef = AA.getModRefInfo(Prev, Loc);
        if (isSend && isModSet(ModRef)) break;
        if (!isSend && isModOrRefSet(ModRef)) break;

        InsertPoint = Prev; 
    }

    if (InsertPoint != CommCall) {
        CommCall->moveBefore(InsertPoint->getIterator());
        return true;
    }
    return false;
}

// Generate individual wait calls
void MDMPPragmaPass::injectWaitsForRegion(Instruction *RegionEnd, AAResults &AA, LLVMContext &Ctx, Module *M) {
    FunctionCallee runtime_wait = M->getOrInsertFunction("mdmp_wait", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));

    for (auto &Req : PendingRequests) {
        Instruction *InsertPt = nullptr;
        
        BasicBlock::iterator it(Req.RuntimeCall);
        it++; 

        // Scan downwards, but strictly within the current Basic Block
        while (it != Req.RuntimeCall->getParent()->end()) {
            Instruction *Next = &*it;
            
            // Did we find the Region End marker?
            if (Next == RegionEnd) {
                InsertPt = RegionEnd;
                break;
            }

            // Did we find a memory dependency (read/write to our buffer)?
            if (isModOrRefSet(AA.getModRefInfo(Next, Req.Loc))) {
                InsertPt = Next;
                break;
            }

            // Did we hit the end of the Basic Block?
            // If we hit a branch (like a jump into a for-loop), we must
            // insert the wait before the branch to guarantee safety.
            if (Next->isTerminator()) {
                InsertPt = Next;
                break;
            }

            it++;
        }

        // Safety fallback (though the terminator check should always catch it)
        if (!InsertPt) InsertPt = RegionEnd; 

        // Inject the wait call
        IRBuilder<> Builder(InsertPt);
        Builder.CreateCall(runtime_wait, {Req.RuntimeCall});
    }
    
    PendingRequests.clear();
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "MDMPPragma", "v0.1",
        [](PassBuilder &PB) {
            // For debugging with 'opt' 
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "mdmp-pragma") {
                        MPM.addPass(MDMPPragmaPass());
                        return true;
                    }
                    return false;
                });

            // For native integration into compilers
            // This injects our pass at the very start of the O1/O2/O3 optimization pipeline
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel Level) {
                    MPM.addPass(MDMPPragmaPass());
                });
        }
    };
}
