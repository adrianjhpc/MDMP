#include "mdmp_pragma_pass.h"

using namespace llvm;

// Structure to hold information about a communication call
struct CommCallInfo {
    CallInst *CI;           // The actual LLVM Call Instruction
    Value *BufferPtr;       // Arg 0: The data pointer
    Value *Count;           // Arg 1: Number of elements
    Value *Type;            // Arg 2: MDMP Type ID
    Value *ByteSize;        // Arg 3: Size in bytes (Crucial for packing!)
    Value *PeerRank;        // Arg 4/5: Dest (for Send) or Src (for Recv)
    Value *Tag;             // Arg 5/6: The message tag
};

PreservedAnalyses MDMPPragmaPass::run(Module &M, ModuleAnalysisManager &MAM) {
    bool changed = false;

    // Grab the Function Analysis Manager proxy
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    // Loop over every function in the module
    for (auto &F : M) {
        if (!F.isDeclaration()) {
            // Grab all three analyses for this specific function
            AAResults &AA = FAM.getResult<AAManager>(F);
            DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
            LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

            // Call runOnFunction so state is properly reset
            changed |= runOnFunction(F, AA, DT, LI);
        }
    }

    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Update the signature to accept the new analyses
bool MDMPPragmaPass::runOnFunction(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI) {
    // Reset the request tracker for the new function
    PendingRequests.clear(); 
    
    // Pass everything down to the transformation engine
    transformPragmasToCalls(F, AA, DT, LI);
    return true;
}

void MDMPPragmaPass::transformPragmasToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI) {
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
    FunctionCallee runtime_wtime = M->getOrInsertFunction("mdmp_wtime", Type::getDoubleTy(Ctx));
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

    FunctionCallee runtime_reduce = M->getOrInsertFunction("mdmp_reduce", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));   
    if (Function *FReduce = dyn_cast<Function>(runtime_reduce.getCallee())) {
        FReduce->addFnAttr(Attribute::NoUnwind);
        // We only mark the in_buf (Arg 0) as ReadOnly. out_buf (Arg 1) is modified
        FReduce->addParamAttr(0, Attribute::ReadOnly);
    }
    FunctionCallee runtime_gather = M->getOrInsertFunction("mdmp_gather", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));   
    if (Function *FGather = dyn_cast<Function>(runtime_gather.getCallee())) {
        FGather->addFnAttr(Attribute::NoUnwind);
        // The send buffer is strictly read-only
        FGather->addParamAttr(0, Attribute::ReadOnly);
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
                hoistInitiation(NewCall, Loc, AA, DT, LI, isSend);
                PendingRequests.push_back({Loc, NewCall});
                toDelete.push_back(CI);
            } 
            else if (Name == "__mdmp_marker_reduce") {
                Value *InBufPtr = CI->getArgOperand(0);
                Value *OutBufPtr = CI->getArgOperand(1);
                Value *CountVal  = CI->getArgOperand(2);
                Value *TypeVal   = CI->getArgOperand(3);
                Value *RootVal   = CI->getArgOperand(5);
                Value *OpVal     = CI->getArgOperand(6);

                CallInst *NewCall = Builder.CreateCall(
                    runtime_reduce, 
                    {InBufPtr, OutBufPtr, CountVal, TypeVal, RootVal, OpVal}
                );
                
                CI->replaceAllUsesWith(NewCall);
                
                // Track the out buffer for data dependencies
                MemoryLocation OutLoc(OutBufPtr, LocationSize::beforeOrAfterPointer());
                
                // Set isSend = false to disable aggressive Loop Hoisting for Collectives
                hoistInitiation(NewCall, OutLoc, AA, DT, LI, false);
                PendingRequests.push_back({OutLoc, NewCall});
                toDelete.push_back(CI);
             }
             else if (Name == "__mdmp_marker_gather") {
                Value *SendBufPtr   = CI->getArgOperand(0);
                Value *SendCountVal = CI->getArgOperand(1);
                Value *RecvBufPtr   = CI->getArgOperand(2);
                Value *TypeVal      = CI->getArgOperand(3); 
                
                Value *RootVal      = CI->getArgOperand(5);

                CallInst *NewCall = Builder.CreateCall(
                    runtime_gather, 
                    {SendBufPtr, SendCountVal, RecvBufPtr, TypeVal, RootVal}
                );
                
                CI->replaceAllUsesWith(NewCall);
                
                // Track the out buffer (recv_buf) for data dependencies
                MemoryLocation OutLoc(RecvBufPtr, LocationSize::beforeOrAfterPointer());
                
                // Disable aggressive loop hoisting for collectives
                hoistInitiation(NewCall, OutLoc, AA, DT, LI, false);
                PendingRequests.push_back({OutLoc, NewCall});
                toDelete.push_back(CI);
            } 
            else if (Name == "__mdmp_marker_commregion_begin") {
                CallInst *NewCall = Builder.CreateCall(runtime_begin);
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_commregion_end") {
                CallInst *NewEnd = Builder.CreateCall(runtime_end);
                injectWaitsForRegion(NewEnd, AA, LI, Ctx, M); // <-- Pass LI here!
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
            else if (Name == "__mdmp_marker_wtime") {
                CallInst *NewCall = Builder.CreateCall(runtime_wtime);
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
        }
    }
    for (Instruction *I : toDelete) I->eraseFromParent();
}

// Move communications to the earliest possible times (helps that they are nonblocking)
void MDMPPragmaPass::hoistInitiation(CallInst *CI, MemoryLocation &Loc, AAResults &AA, DominatorTree &DT, LoopInfo &LI, bool isSend) {
    // 1. === LOOP INVARIANT CODE MOTION (LICM) ===
    Loop *L = LI.getLoopFor(CI->getParent());
    
    if (L && isSend) {
        bool isSafeToHoistOut = true;

        // NEW DEPENDENCY CHECK: Are any of our arguments calculated inside this loop?
        // If so, we cannot hoist the SEND outside the loop!
        for (Value *Op : CI->operands()) {
            if (Instruction *OpInst = dyn_cast<Instruction>(Op)) {
                if (L->contains(OpInst)) {
                    isSafeToHoistOut = false;
                    break;
                }
            }
        }
        
        if (isSafeToHoistOut) {
            for (BasicBlock *BB : L->blocks()) {
                for (Instruction &I : *BB) {
                    // BARRIER CHECK
                    if (auto *Call = dyn_cast<CallInst>(&I)) {
                        if (Call->getCalledFunction() && Call->getCalledFunction()->getName() == "mdmp_commregion_begin") {
                            isSafeToHoistOut = false; 
                            break; 
                        }
                    }
                    // MEMORY CHECK
                    if (I.mayWriteToMemory() && isModSet(AA.getModRefInfo(&I, Loc))) {
                        isSafeToHoistOut = false;
                        break;
                    }
                }
                if (!isSafeToHoistOut) break;
            }
        }

        // TELEPORTATION
        if (isSafeToHoistOut) {
            BasicBlock *Preheader = L->getLoopPreheader();
            if (Preheader) {
                CI->moveBefore(Preheader->getTerminator()->getIterator());
                errs() << "[MDMP] OPTIMIZATION: Hoisted SEND out of loop!\n";
                return; 
            }
        }
    }

    // 2. === STANDARD LINEAR HOISTING ===
    Instruction *InsertPoint = CI;
    Instruction *Prev = CI->getPrevNode();

    while (Prev) {
        // NEW RULE 1: Never move above a PHI Node!
        if (isa<PHINode>(Prev)) break;

        // NEW RULE 2: Never move above the instruction that calculates our arguments!
        bool usesPrev = false;
        for (Value *Op : CI->operands()) {
            if (Op == Prev) {
                usesPrev = true;
                break;
            }
        }
        if (usesPrev) break;

        // BARRIER CHECK
        if (auto *Call = dyn_cast<CallInst>(Prev)) {
            if (Call->getCalledFunction() && Call->getCalledFunction()->getName() == "mdmp_commregion_begin") {
                break; 
            }
        }

        // MEMORY CHECK
        if (Prev->mayWriteToMemory() && isModSet(AA.getModRefInfo(Prev, Loc))) {
            break; 
        }

        InsertPoint = Prev;
        Prev = Prev->getPrevNode();
    }

    // Apply the hoist
    if (InsertPoint != CI) {
        CI->moveBefore(InsertPoint->getIterator());
        errs() << "[MDMP] Hoisted communication within region.\n";
    }
}

// Generate individual wait calls dynamically based on CFG Dataflow Analysis
void MDMPPragmaPass::injectWaitsForRegion(Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, LLVMContext &Ctx, Module *M){
    FunctionCallee runtime_wait = M->getOrInsertFunction("mdmp_wait", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));

    for (auto &Req : PendingRequests) {
        SmallVector<Instruction*, 4> WaitInsertionPoints;
        SmallPtrSet<BasicBlock*, 8> Visited;
        
        struct TraversalState {
            BasicBlock *BB;
            BasicBlock::iterator StartIt;
        };
        SmallVector<TraversalState, 8> Worklist;
        
        // Start exploring from the instruction immediately following the async call
        BasicBlock::iterator StartIt = Req.RuntimeCall->getIterator();
        StartIt++;
        Worklist.push_back({Req.RuntimeCall->getParent(), StartIt});
        
        while (!Worklist.empty()) {
            auto State = Worklist.pop_back_val();
            BasicBlock *BB = State.BB;
            
            // If we've already fully visited this block from the top, skip it to prevent infinite loops
            if (!Visited.insert(BB).second && State.StartIt == BB->begin()) {
                continue;
            }
            
            bool foundWaitPoint = false;
            for (auto It = State.StartIt; It != BB->end(); ++It) {
                Instruction *Inst = &*It;
                
                // Stop Condition 1: We reached the explicit end of the region
                if (Inst == RegionEnd) {
                    WaitInsertionPoints.push_back(Inst);
                    foundWaitPoint = true;
                    break;
                }
                
                // Stop Condition 2: Memory dependency (Read or Write to the buffer)
                if (Inst->mayReadOrWriteMemory()) {
                    
                    bool isAsyncCall = false;
                    if (auto *Call = dyn_cast<CallInst>(Inst)) {
                        if (Function *CalledFn = Call->getCalledFunction()) {
                            StringRef FnName = CalledFn->getName();
                            if (FnName == "mdmp_send" || FnName == "mdmp_recv" || 
                                FnName == "mdmp_reduce" || FnName == "mdmp_gather" ||
                                FnName.starts_with("__mdmp_marker_")) {
                                isAsyncCall = true;
                            }
                        }
                    }
                    
                    // Bypass AA's 'Unknown Size' panic by proving the instruction 
                    // accesses a fundamentally different base object than our buffer.
                    bool isSafeMemOp = false;
                    if (auto *LI = dyn_cast<LoadInst>(Inst)) {
                        if (getUnderlyingObject(LI->getPointerOperand()) != getUnderlyingObject(Req.Loc.Ptr)) {
                            isSafeMemOp = true; // Safely loading a different variable
                        }
                    } else if (auto *SI = dyn_cast<StoreInst>(Inst)) {
                        if (getUnderlyingObject(SI->getPointerOperand()) != getUnderlyingObject(Req.Loc.Ptr)) {
                            isSafeMemOp = true; // Safely storing to a different variable
                        }
                    }
                    
                    // Only drop the wait if it's not a whitelisted async call, 
                    // not a safe variable operation, and AA claims a collision.
                    if (!isAsyncCall && !isSafeMemOp && isModOrRefSet(AA.getModRefInfo(Inst, Req.Loc))) {
                        WaitInsertionPoints.push_back(Inst);
                        foundWaitPoint = true;
                        break;
                    }
                }
            }
            // If we reached the end of the block without finding a reason to wait,
            // we jump across the terminator and explore all successor blocks
            if (!foundWaitPoint) {
                for (BasicBlock *Succ : successors(BB)) {
                    // If the successor block is deeper inside a loop than our Send/Recv,
                    // we must not enter it, or we will poison the inner compute math.
                    // We drop the wait safely at the terminator before the loop starts.
                    if (LI.getLoopDepth(Succ) > LI.getLoopDepth(Req.RuntimeCall->getParent())) {
                        WaitInsertionPoints.push_back(BB->getTerminator());
                    } else {
                        Worklist.push_back({Succ, Succ->begin()});
                    }
                }
            }
        }
        
        // Insert the waits at the discovered optimal points
        // We use a set to ensure we don't insert duplicate waits on the exact same instruction 
        // if multiple execution paths converge.
        SmallPtrSet<Instruction*, 4> UniqueWaitPoints(WaitInsertionPoints.begin(), WaitInsertionPoints.end());
        for (Instruction *InsertPt : UniqueWaitPoints) {
            IRBuilder<> Builder(InsertPt);
            // Note: Req.RuntimeCall returns the integer 'req_id', which we pass directly to mdmp_wait
            Builder.CreateCall(runtime_wait, {Req.RuntimeCall});
        }
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
