#include "mdmp_compiler_pass.h"
#include "llvm/Analysis/ValueTracking.h" 
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

struct TrackedBuffer {
    MemoryLocation Loc;
    bool isNetworkReadOnly; // True = Send buffer (CPU reads ok), False = Recv buffer (CPU reads/writes collide)
};

struct AsyncRequest {
    std::vector<TrackedBuffer> Buffers;
    CallInst *RuntimeCall;
};

// Global tracker for the current function being processed
std::vector<AsyncRequest> PendingRequests;

PreservedAnalyses MDMPPass::run(Module &M, ModuleAnalysisManager &MAM) {
    bool changed = false;
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    for (auto &F : M) {
        if (!F.isDeclaration()) {
            AAResults &AA = FAM.getResult<AAManager>(F);
            DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
            LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
            changed |= runOnFunction(F, AA, DT, LI);
        }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}


bool MDMPPass::runOnFunction(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI) {
    PendingRequests.clear();
    
    transformFunctionsToCalls(F, AA, DT, LI);
    
    // If the function ends and we still have pending imperative calls 
    // that weren't part of a CommRegion, trace and inject their waits now!
    if (!PendingRequests.empty()) {
        LLVMContext &Ctx = F.getContext();
        Module *M = F.getParent();
        injectWaitsForRegion(nullptr, AA, LI, Ctx, M, DT);
    }
    
    return true;
}

void MDMPPass::transformFunctionsToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    // Utility functions
    FunctionCallee runtime_begin = M->getOrInsertFunction("mdmp_commregion_begin", Type::getVoidTy(Ctx));
    FunctionCallee runtime_end   = M->getOrInsertFunction("mdmp_commregion_end", Type::getVoidTy(Ctx));
    FunctionCallee runtime_sync  = M->getOrInsertFunction("mdmp_sync", Type::getVoidTy(Ctx));
    FunctionCallee runtime_init  = M->getOrInsertFunction("mdmp_init", Type::getVoidTy(Ctx));
    FunctionCallee runtime_final = M->getOrInsertFunction("mdmp_final", Type::getVoidTy(Ctx));
    FunctionCallee runtime_get_rank = M->getOrInsertFunction("mdmp_get_rank", Type::getInt32Ty(Ctx));
    FunctionCallee runtime_get_size = M->getOrInsertFunction("mdmp_get_size", Type::getInt32Ty(Ctx));
    FunctionCallee runtime_wtime = M->getOrInsertFunction("mdmp_wtime", Type::getDoubleTy(Ctx));
    FunctionCallee runtime_set_debug = M->getOrInsertFunction("mdmp_set_debug", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));
    FunctionCallee runtime_abort = M->getOrInsertFunction("mdmp_abort", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));

 
    // Imperative functionality apis (Returns i32 Req ID) 
    FunctionCallee runtime_send = M->getOrInsertFunction("mdmp_send", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    if (Function *FSend = dyn_cast<Function>(runtime_send.getCallee())) {
        FSend->addFnAttr(Attribute::NoUnwind); FSend->setMemoryEffects(MemoryEffects::readOnly()); FSend->addParamAttr(0, Attribute::ReadOnly);
    }
 
    FunctionCallee runtime_recv = M->getOrInsertFunction("mdmp_recv", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    if (Function *FRecv = dyn_cast<Function>(runtime_recv.getCallee())) { FRecv->addFnAttr(Attribute::NoUnwind); }

    // ==========================================
    // Imperative Collective apis
    // ==========================================
    FunctionCallee runtime_reduce = M->getOrInsertFunction("mdmp_reduce", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)); 
    if (Function *FReduce = dyn_cast<Function>(runtime_reduce.getCallee())) { FReduce->addFnAttr(Attribute::NoUnwind); }

    FunctionCallee runtime_gather = M->getOrInsertFunction("mdmp_gather", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx)); 
    if (Function *FGather = dyn_cast<Function>(runtime_gather.getCallee())) { FGather->addFnAttr(Attribute::NoUnwind); }

    FunctionCallee runtime_allreduce = M->getOrInsertFunction("mdmp_allreduce", 
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx)); 
    if (Function *FAllreduce = dyn_cast<Function>(runtime_allreduce.getCallee())) { FAllreduce->addFnAttr(Attribute::NoUnwind); }
    
    FunctionCallee runtime_allgather = M->getOrInsertFunction("mdmp_allgather",
        Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx));
    if (Function *FAllgather = dyn_cast<Function>(runtime_allgather.getCallee())) { FAllgather->addFnAttr(Attribute::NoUnwind); }


    // ==========================================
    // Declarative Collective apis
    // ==========================================
    FunctionCallee runtime_register_send = M->getOrInsertFunction("mdmp_register_send", 
        Type::getVoidTy(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    if (Function *FRegSend = dyn_cast<Function>(runtime_register_send.getCallee())) { FRegSend->addFnAttr(Attribute::NoUnwind); }
        
    FunctionCallee runtime_register_recv = M->getOrInsertFunction("mdmp_register_recv", 
        Type::getVoidTy(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    if (Function *FRegRecv = dyn_cast<Function>(runtime_register_recv.getCallee())) { FRegRecv->addFnAttr(Attribute::NoUnwind); }

    FunctionCallee runtime_register_reduce = M->getOrInsertFunction("mdmp_register_reduce", 
        Type::getVoidTy(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)); 
    if (Function *FRegReduce = dyn_cast<Function>(runtime_register_reduce.getCallee())) { FRegReduce->addFnAttr(Attribute::NoUnwind); }
        
    FunctionCallee runtime_register_gather = M->getOrInsertFunction("mdmp_register_gather", 
        Type::getVoidTy(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx));
    if (Function *FRegGather = dyn_cast<Function>(runtime_register_gather.getCallee())) { FRegGather->addFnAttr(Attribute::NoUnwind); }

    FunctionCallee runtime_register_allreduce = M->getOrInsertFunction("mdmp_register_allreduce",
        Type::getVoidTy(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx)); 
    if (Function *FRegAllreduce = dyn_cast<Function>(runtime_register_allreduce.getCallee())) { FRegAllreduce->addFnAttr(Attribute::NoUnwind); }

    FunctionCallee runtime_register_allgather = M->getOrInsertFunction("mdmp_register_allgather",
        Type::getVoidTy(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
        PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)); 
    if (Function *FRegAllgather = dyn_cast<Function>(runtime_register_allgather.getCallee())) { FRegAllgather->addFnAttr(Attribute::NoUnwind); }

    FunctionCallee runtime_commit = M->getOrInsertFunction("mdmp_commit", Type::getInt32Ty(Ctx));

    std::vector<Instruction*> toDelete;
    std::vector<TrackedBuffer> ActiveRegionLocs; // Tracks buffers for the Declarative paradigm
    std::vector<MemoryLocation> ActiveRegionLocs; 

    for (auto &BB : F) {
        for (auto &I : BB) {
            auto *CI = dyn_cast<CallInst>(&I);
            if (!CI || !CI->getCalledFunction()) continue;
            
            StringRef Name = CI->getCalledFunction()->getName();
            IRBuilder<> Builder(CI);

            if (Name == "__mdmp_marker_commregion_begin") {
                CallInst *NewCall = Builder.CreateCall(runtime_begin);
                CI->replaceAllUsesWith(NewCall);
                ActiveRegionLocs.clear(); 
                toDelete.push_back(CI);
            }
            // ==========================================
            // Imperative functions (sends/recvs)
            // ==========================================
            else if (Name == "__mdmp_marker_send" || Name == "__mdmp_marker_recv") {
                Value *BufferPtr = CI->getArgOperand(0);
                Value *CountVal  = CI->getArgOperand(1);
                Value *TypeVal   = CI->getArgOperand(2);
                Value *ByteSize  = CI->getArgOperand(3);
                Value *ActorRank = CI->getArgOperand(4);
                Value *PeerRank  = CI->getArgOperand(5);
                Value *TagVal    = CI->getArgOperand(6); 

                LocationSize LocSize = LocationSize::beforeOrAfterPointer();
                if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) {
                    LocSize = LocationSize::precise(ConstBytes->getZExtValue());
                }
                std::vector<MemoryLocation> Locs = { MemoryLocation(BufferPtr, LocSize) };

		bool isSend = (Name == "__mdmp_marker_send");
                std::vector<TrackedBuffer> TrackedLocs = { {MemoryLocation(BufferPtr, LocSize), isSend} };
                
                CallInst *NewCall = Builder.CreateCall(isSend ? runtime_send : runtime_recv, 
                    {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});
                
                CI->replaceAllUsesWith(NewCall);
                
                hoistInitiation(NewCall, TrackedLocs, AA, DT, LI, isSend);
                PendingRequests.push_back({TrackedLocs, NewCall});
                toDelete.push_back(CI);
            } 
            // ==========================================
            // Declarative functionality (register/commit)
            // ==========================================
            else if (Name == "__mdmp_marker_register_send" || Name == "__mdmp_marker_register_recv") {
                Value *BufferPtr = CI->getArgOperand(0);
                Value *CountVal  = CI->getArgOperand(1);
                Value *TypeVal   = CI->getArgOperand(2);
                Value *ByteSize  = CI->getArgOperand(3);
                Value *ActorRank = CI->getArgOperand(4);
                Value *PeerRank  = CI->getArgOperand(5);
                Value *TagVal    = CI->getArgOperand(6); 

                LocationSize LocSize = LocationSize::beforeOrAfterPointer();
                if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
                ActiveRegionLocs.push_back(MemoryLocation(BufferPtr, LocSize));

		bool isSend = (Name == "__mdmp_marker_register_send");
                ActiveRegionLocs.push_back({MemoryLocation(BufferPtr, LocSize), isSend});
                
                CallInst *NewCall = Builder.CreateCall(isSend ? runtime_register_send : runtime_register_recv, 
                    {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});
                
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_register_reduce") {
                Value *InBuf = CI->getArgOperand(0); Value *OutBuf = CI->getArgOperand(1);
                Value *ByteSize = CI->getArgOperand(4); 

                LocationSize LocSize = LocationSize::beforeOrAfterPointer();
                if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
                
                // Track both buffers to protect them
                ActiveRegionLocs.push_back(MemoryLocation(InBuf, LocSize));
                ActiveRegionLocs.push_back(MemoryLocation(OutBuf, LocSize));

                CallInst *NewCall = Builder.CreateCall(runtime_register_reduce, 
                    {InBuf, OutBuf, CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_register_gather") {
                Value *SendBuf = CI->getArgOperand(0); Value *RecvBuf = CI->getArgOperand(2);
                Value *ByteSize = CI->getArgOperand(4); 

                LocationSize LocSize = LocationSize::beforeOrAfterPointer();
                if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
                
                // Track both buffers to protect them
                ActiveRegionLocs.push_back(MemoryLocation(SendBuf, LocSize));
                ActiveRegionLocs.push_back(MemoryLocation(RecvBuf, LocSize));

                CallInst *NewCall = Builder.CreateCall(runtime_register_gather, 
                    {SendBuf, CI->getArgOperand(1), RecvBuf, CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            } 
            else if (Name == "__mdmp_marker_commit") {
                CallInst *NewCommit = Builder.CreateCall(runtime_commit);
                CI->replaceAllUsesWith(NewCommit);
                
                // Hoist commits to overall user work with the coalesced network payload
                hoistInitiation(NewCommit, ActiveRegionLocs, AA, DT, LI, true);
                PendingRequests.push_back({ActiveRegionLocs, NewCommit});
                
                ActiveRegionLocs.clear(); // Reset in case the programmer has called commit multiple times
                toDelete.push_back(CI);
            }
            // ==========================================
            // Collective operations
            // ==========================================
            else if (Name == "__mdmp_marker_reduce") {
                CallInst *NewCall = Builder.CreateCall(runtime_reduce, 
                    {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
                CI->replaceAllUsesWith(NewCall);
                
                std::vector<MemoryLocation> Locs = {
                    MemoryLocation(CI->getArgOperand(1), LocationSize::beforeOrAfterPointer()), // Out Buf
                    MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer())  // In Buf
                };
                hoistInitiation(NewCall, Locs, AA, DT, LI, false);
                PendingRequests.push_back({Locs, NewCall});
                toDelete.push_back(CI);
             }
             else if (Name == "__mdmp_marker_gather") {
                CallInst *NewCall = Builder.CreateCall(runtime_gather, 
                    {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
                CI->replaceAllUsesWith(NewCall);
                
                std::vector<MemoryLocation> Locs = {
                    MemoryLocation(CI->getArgOperand(2), LocationSize::beforeOrAfterPointer()), // Recv Buf
                    MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer())  // Send Buf
                };
                hoistInitiation(NewCall, Locs, AA, DT, LI, false);
                PendingRequests.push_back({Locs, NewCall});
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_allreduce" || Name == "__mdmp_marker_register_allreduce") {
                FunctionCallee target_func = (Name == "__mdmp_marker_allreduce") ? runtime_allreduce : runtime_register_allreduce;
                CallInst *NewCall = Builder.CreateCall(target_func, 
                    {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
                CI->replaceAllUsesWith(NewCall);
                
                std::vector<MemoryLocation> Locs = {
                    MemoryLocation(CI->getArgOperand(1), LocationSize::beforeOrAfterPointer()), // Out Buf
                    MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer())  // In Buf
                };

                if (Name == "__mdmp_marker_allreduce") {
                    hoistInitiation(NewCall, Locs, AA, DT, LI, false); 
                    AsyncRequest Req;
                    Req.RuntimeCall = NewCall;
                    Req.Locs = Locs;
                    PendingRequests.push_back(Req);
                    llvm::errs() << "[MDMP PASS DEBUG] Caught Imperative Allreduce.\n";
                } else {
                    ActiveRegionLocs.push_back(Locs[0]);
                    ActiveRegionLocs.push_back(Locs[1]);
                }
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_allgather" || Name == "__mdmp_marker_register_allgather") {
                FunctionCallee target_func = (Name == "__mdmp_marker_allgather") ? runtime_allgather : runtime_register_allgather;
                CallInst *NewCall = Builder.CreateCall(target_func, 
                    {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});
                CI->replaceAllUsesWith(NewCall);
                
                std::vector<MemoryLocation> Locs = {
                    MemoryLocation(CI->getArgOperand(2), LocationSize::beforeOrAfterPointer()), // Out Buf
                    MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer())  // In Buf
                };

                if (Name == "__mdmp_marker_allgather") {
                    hoistInitiation(NewCall, Locs, AA, DT, LI, false);
                    AsyncRequest Req;
                    Req.RuntimeCall = NewCall;
                    Req.Locs = Locs;
                    PendingRequests.push_back(Req);
                    llvm::errs() << "[MDMP PASS DEBUG] Caught Imperative Allgather.\n";
                } else {
                    ActiveRegionLocs.push_back(Locs[0]);
                    ActiveRegionLocs.push_back(Locs[1]);
                }
                toDelete.push_back(CI);
            } 
            else if (Name == "__mdmp_marker_commregion_end") {
                CallInst *NewEnd = Builder.CreateCall(runtime_end);
                injectWaitsForRegion(NewEnd, AA, LI, Ctx, M, DT);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_get_rank") { CallInst *NewCall = Builder.CreateCall(runtime_get_rank); CI->replaceAllUsesWith(NewCall); toDelete.push_back(CI); }
            else if (Name == "__mdmp_marker_get_size") { CallInst *NewCall = Builder.CreateCall(runtime_get_size); CI->replaceAllUsesWith(NewCall); toDelete.push_back(CI); }
            else if (Name == "__mdmp_marker_init") { Builder.CreateCall(runtime_init); toDelete.push_back(CI); }
            else if (Name == "__mdmp_marker_final") { Builder.CreateCall(runtime_final); toDelete.push_back(CI); }
            else if (Name == "__mdmp_marker_sync") { Builder.CreateCall(runtime_sync); toDelete.push_back(CI); }
            else if (Name == "__mdmp_marker_wtime") { CallInst *NewCall = Builder.CreateCall(runtime_wtime); CI->replaceAllUsesWith(NewCall); toDelete.push_back(CI); }
            else if (Name == "__mdmp_marker_set_debug") {
                Value *EnableArg = CI->getArgOperand(0); 
                CallInst *NewCall = Builder.CreateCall(runtime_set_debug, {EnableArg});
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
            else if (Name == "__mdmp_marker_abort") {
                Value *EnableArg = CI->getArgOperand(0);
                CallInst *NewCall = Builder.CreateCall(runtime_abort, {EnableArg});
                CI->replaceAllUsesWith(NewCall);
                toDelete.push_back(CI);
            }
        }
    }
    for (Instruction *I : toDelete) I->eraseFromParent();

}

void MDMPPass::hoistInitiation(CallInst *CI, std::vector<MemoryLocation> &Locs, AAResults &AA, DominatorTree &DT, LoopInfo &LI, bool isSend) {
    Loop *L = LI.getLoopFor(CI->getParent());
    if (L && isSend) {
        bool isSafeToHoistOut = true;
        for (Value *Op : CI->operands()) {
            if (Instruction *OpInst = dyn_cast<Instruction>(Op)) {
                if (L->contains(OpInst)) { isSafeToHoistOut = false; break; }
            }
        }
        if (isSafeToHoistOut) {
            for (BasicBlock *BB : L->blocks()) {
                for (Instruction &I : *BB) {
                    if (auto *Call = dyn_cast<CallInst>(&I)) {
                        if (Call->getCalledFunction() && Call->getCalledFunction()->getName() == "mdmp_commregion_begin") { isSafeToHoistOut = false; break; }
                    }
                    if (I.mayWriteToMemory()) {
                        for (auto &Loc : Locs) {
                            if (isModSet(AA.getModRefInfo(&I, Loc))) { isSafeToHoistOut = false; break; }
                        }
                    }
                }
                if (!isSafeToHoistOut) break;
            }
        }
        if (isSafeToHoistOut) {
            BasicBlock *Preheader = L->getLoopPreheader();
            if (Preheader) {
                CI->moveBefore(Preheader->getTerminator()->getIterator());
                return; 
            }
        }
    }

    Instruction *InsertPoint = CI;
    Instruction *Prev = CI->getPrevNode();
    while (Prev) {
        if (isa<PHINode>(Prev)) break;
        bool usesPrev = false;
        for (Value *Op : CI->operands()) { if (Op == Prev) { usesPrev = true; break; } }
        if (usesPrev) break;
        
        if (auto *Call = dyn_cast<CallInst>(Prev)) {
            if (Call->getCalledFunction() && Call->getCalledFunction()->getName() == "mdmp_commregion_begin") break; 
        }
        
        bool memoryCollision = false;
        
        if (Prev->mayWriteToMemory() || (!isSend && Prev->mayReadFromMemory())) {
            
            // Extreme Exact Match (Bypasses TBAA Blindspots)
            if (auto *SI = dyn_cast<StoreInst>(Prev)) {
                MemoryLocation AccessedLoc = MemoryLocation::get(SI);
                for (auto &Loc : Locs) {
                    if (AccessedLoc.Ptr->stripPointerCasts() == Loc.Ptr->stripPointerCasts() ||
                        getUnderlyingObject(AccessedLoc.Ptr) == getUnderlyingObject(Loc.Ptr) ||
                        AA.alias(AccessedLoc, Loc) != llvm::AliasResult::NoAlias) {
                        memoryCollision = true; break;
                    }
                }
            } else if (auto *LI = dyn_cast<LoadInst>(Prev)) {
                MemoryLocation AccessedLoc = MemoryLocation::get(LI);
                for (auto &Loc : Locs) {
                    if (AccessedLoc.Ptr->stripPointerCasts() == Loc.Ptr->stripPointerCasts() ||
                        getUnderlyingObject(AccessedLoc.Ptr) == getUnderlyingObject(Loc.Ptr) ||
                        AA.alias(AccessedLoc, Loc) != llvm::AliasResult::NoAlias) {
                        memoryCollision = true; break;
                    }
                }
            } else {
                // Fallback to AA for Opaque calls/Intrinsics
                for (auto &Loc : Locs) {
                    if (isModSet(AA.getModRefInfo(Prev, Loc))) { 
                        memoryCollision = true; break; 
                    }
                    if (!isSend && isModOrRefSet(AA.getModRefInfo(Prev, Loc))) {
                        memoryCollision = true; break;
                    }
                }
            }
        }
        
        // If we hit a collision, we must immediately stop pulling the network call backward
        if (memoryCollision) break; 
        
        InsertPoint = Prev;
        Prev = Prev->getPrevNode();
    }
    if (InsertPoint != CI) { CI->moveBefore(InsertPoint->getIterator()); }
}

void MDMPPass::injectWaitsForRegion(Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, LLVMContext &Ctx, Module *M, DominatorTree &DT) {
    FunctionCallee runtime_wait = M->getOrInsertFunction("mdmp_wait", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));

    struct TraversalState { BasicBlock *BB; BasicBlock::iterator StartIt; };

    SmallVector<Instruction*, 16> WaitInsertionPoints;
    SmallPtrSet<BasicBlock*, 16> Visited;
    SmallVector<TraversalState, 16> Worklist;

    for (auto &Req : PendingRequests) {
        WaitInsertionPoints.clear();
        Visited.clear();
        Worklist.clear();

        SmallVector<TraversalState, 8> Worklist;
        
        BasicBlock::iterator StartIt = Req.RuntimeCall->getIterator();
        StartIt++;
        Worklist.push_back({Req.RuntimeCall->getParent(), StartIt});
        
        while (!Worklist.empty()) {
            auto State = Worklist.pop_back_val();
            BasicBlock *BB = State.BB;
            
            if (!Visited.insert(BB).second && State.StartIt == BB->begin()) continue;
            
            bool foundWaitPoint = false;
            for (auto It = State.StartIt; It != BB->end(); ++It) {
                Instruction *Inst = &*It;
                
                if (Inst == RegionEnd) {
                    WaitInsertionPoints.push_back(Inst); foundWaitPoint = true; break;
                }
                
                if (Inst->mayReadOrWriteMemory()) {
                    bool isAsyncCall = false;
                    if (auto *Call = dyn_cast<CallInst>(Inst)) {
                        if (Function *CalledFn = Call->getCalledFunction()) {
                            StringRef FnName = CalledFn->getName();
                            if (FnName == "mdmp_send" || FnName == "mdmp_recv" || FnName == "mdmp_reduce" || 
                                FnName == "mdmp_allreduce" || FnName == "mdmp_allgather" || 
                                FnName == "mdmp_gather" || FnName == "mdmp_commit" || 
                                FnName == "mdmp_wait" || FnName == "mdmp_wtime" || // <-- ADDED THESE TWO
                                FnName == "mdmp_register_send" || FnName == "mdmp_register_recv" ||
                                FnName == "mdmp_register_reduce" || FnName == "mdmp_register_gather" ||
                                FnName == "mdmp_register_allreduce" || FnName == "mdmp_register_allgather" ||
                                FnName.starts_with("__mdmp_marker_")) {
                                isAsyncCall = true;
                            }
                        }
                    }
                    
                    // Manual check for exact pointer matches (bypasses AA limitations)
                    bool exactMatch = false;
                    if (auto *LInst = dyn_cast<LoadInst>(Inst)) {
                        Value *LoadPtr = LInst->getPointerOperand()->stripPointerCasts();
                        for (auto &Loc : Req.Locs) {
                            if (LoadPtr == Loc.Ptr->stripPointerCasts() || 
                                getUnderlyingObject(LoadPtr) == getUnderlyingObject(Loc.Ptr)) { 
                                exactMatch = true; break; 
                            }
                        }
                    } else if (auto *SInst = dyn_cast<StoreInst>(Inst)) {
                        Value *StorePtr = SInst->getPointerOperand()->stripPointerCasts();
                        for (auto &Loc : Req.Locs) {
                            if (StorePtr == Loc.Ptr->stripPointerCasts() || 
                                getUnderlyingObject(StorePtr) == getUnderlyingObject(Loc.Ptr)) { 
                                exactMatch = true; break; 
                            }
                        }
                    }

		    // Collision Router
                    bool memoryCollision = false;

                    // Find any hard barriers we have
                    if (auto *Call = dyn_cast<CallInst>(Inst)) {
                        if (Function *CalledFn = Call->getCalledFunction()) {
                            StringRef Name = CalledFn->getName();
                            if (Name == "mdmp_final" || Name == "__mdmp_marker_final" || 
                                Name == "MPI_Finalize" || Name == "mdmp_abort" || Name == "__mdmp_marker_abort") {
                                llvm::errs() << "[MDMP PASS DEBUG] Collision triggered by Finalize/Abort Barrier.\n";
                                memoryCollision = true;
                            }
                        }
                    }

                    if (!memoryCollision && !isAsyncCall) {
                        
                        // Track loading and storing
                        if (auto *LI = dyn_cast<LoadInst>(Inst)) {
                            MemoryLocation AccessedLoc = MemoryLocation::get(LI);
                            for (auto &Buf : Req.Buffers) {
                                // Reads from a send buffer are ok
                                if (!Buf.isNetworkReadOnly) { 
                                    if (AccessedLoc.Ptr->stripPointerCasts() == Buf.Loc.Ptr->stripPointerCasts() ||
                                        getUnderlyingObject(AccessedLoc.Ptr) == getUnderlyingObject(Buf.Loc.Ptr) ||
                                        AA.alias(AccessedLoc, Buf.Loc) != llvm::AliasResult::NoAlias) {
                                        memoryCollision = true; break;
                                    }
                                }
                            }
                        } 
                        else if (auto *SI = dyn_cast<StoreInst>(Inst)) {
                            MemoryLocation AccessedLoc = MemoryLocation::get(SI);
                            for (auto &Buf : Req.Buffers) {
                                // CPU writes always collide, whether it's a Send or Recv buffer
                                if (AccessedLoc.Ptr->stripPointerCasts() == Buf.Loc.Ptr->stripPointerCasts() ||
                                    getUnderlyingObject(AccessedLoc.Ptr) == getUnderlyingObject(Buf.Loc.Ptr) ||
                                    AA.alias(AccessedLoc, Buf.Loc) != llvm::AliasResult::NoAlias) {
                                    memoryCollision = true; break;
                                }
                            }
                        }
                        // Capture opaque function calls
                        else if (auto *Call = dyn_cast<CallInst>(Inst)) {
                            StringRef FnName = "";
                            if (Call->getCalledFunction()) FnName = Call->getCalledFunction()->getName();
                            
                            bool isHarmlessSetup = FnName.starts_with("_ZNSt6vector") || 
                                                   FnName.starts_with("_ZNKSt6vector") ||
                                                   FnName.contains("dataEv") || 
                                                   FnName.contains("sizeEv") || 
                                                   FnName.contains("swap");
                            
                            if (!isHarmlessSetup) {
                                for (Value *Arg : Call->args()) {
                                    for (auto &Buf : Req.Buffers) {
                                        if (Arg->stripPointerCasts() == Buf.Loc.Ptr->stripPointerCasts() ||
                                            getUnderlyingObject(Arg) == getUnderlyingObject(Buf.Loc.Ptr)) {
                                            memoryCollision = true; break;
                                        }
                                    }
                                    if (memoryCollision) break;
                                }
                                
                                // Alias Analysis fallback for complex C++ calls
                                if (!memoryCollision) {
                                    for (auto &Buf : Req.Buffers) {
                                        auto MR = AA.getModRefInfo(Call, Buf.Loc);
                                        if (Buf.isNetworkReadOnly && isModSet(MR)) { memoryCollision = true; break; }
                                        else if (!Buf.isNetworkReadOnly && isModOrRefSet(MR)) { memoryCollision = true; break; }
                                    }
                                }
                            }
                        }
                        // Fallback for other memory instructions
                        else if (Inst->mayReadOrWriteMemory()) {
                            for (auto &Buf : Req.Buffers) {
                                auto MR = AA.getModRefInfo(Inst, Buf.Loc);
                                if (Buf.isNetworkReadOnly && isModSet(MR)) { memoryCollision = true; break; }
                                else if (!Buf.isNetworkReadOnly && isModOrRefSet(MR)) { memoryCollision = true; break; }
                            }
                        }
                    }
                    if (memoryCollision) {
                        WaitInsertionPoints.push_back(Inst);
                        foundWaitPoint = true;
                        break; 
                    }
                }
            } 
            if (!foundWaitPoint) {
                bool hasSuccessors = false;
                Loop *ReqLoop = LI.getLoopFor(Req.RuntimeCall->getParent());

                for (BasicBlock *Succ : successors(BB)) {
                    hasSuccessors = true;
                    
                    bool isBackedge = false;
                    Loop *EdgeLoop = LI.getLoopFor(BB);
                    while (EdgeLoop) {
                        if (EdgeLoop->getHeader() == Succ) {
                            isBackedge = true;
                            break;
                        }
                        EdgeLoop = EdgeLoop->getParentLoop();
                    }

                    if (isBackedge && EdgeLoop) {
                        // If we cross the backedge of the exact loop where the request was made 
                        // (e.g., dir=0..4), allow it to wrap so we don't serialize neighbor exchanges
                        // However if we cross the backedge of an outer loop (e.g., the timestep),
                        // we must force a wait to prevent flooding the network.
                        if (!ReqLoop || (EdgeLoop != ReqLoop && EdgeLoop->contains(ReqLoop))) {
                            llvm::errs() << "[MDMP PASS DEBUG] Wait forced at Outer Loop Backedge.\n";
                            WaitInsertionPoints.push_back(BB->getTerminator());
                            continue; 
                        }
                    }

                    // Safe to keep exploring forward
                    Worklist.push_back({Succ, Succ->begin()});
                }
                
                if (!hasSuccessors) {
                    llvm::errs() << "[MDMP PASS DEBUG] Wait forced at End of Function.\n";
                    WaitInsertionPoints.push_back(BB->getTerminator());
                }
            } 
        }
        
        SmallPtrSet<Instruction*, 4> UniqueWaitPoints;
        for (Instruction *InsertPt : WaitInsertionPoints) {
            
            // If the collision happens in a block not mathematically dominated by the network request,
            // we force the wait to happen immediately before the flow leaves the safe dominated block
            if (!DT.dominates(Req.RuntimeCall, InsertPt)) {
                InsertPt = Req.RuntimeCall->getParent()->getTerminator();
            }

            Instruction *HoistPt = InsertPt;
            Loop *L = LI.getLoopFor(HoistPt->getParent());
            Loop *ReqLoop = LI.getLoopFor(Req.RuntimeCall->getParent());

            while (L && L != ReqLoop) {
                BasicBlock *Latch = L->getLoopLatch();
                if (Latch && DT.dominates(HoistPt->getParent(), Latch)) {
                    BasicBlock *Preheader = L->getLoopPreheader();
                    if (Preheader) {
                        Instruction *PotentialHoistPt = Preheader->getTerminator();
                        // Never hoist the Wait instruction above the actual Request creation
                        if (DT.dominates(Req.RuntimeCall, PotentialHoistPt)) {
                            HoistPt = PotentialHoistPt;
                            L = LI.getLoopFor(HoistPt->getParent());
                            continue;
                        }
                    }
                }
                break; 
            }
            UniqueWaitPoints.insert(HoistPt);
        }

        for (Instruction *InsertPt : UniqueWaitPoints) {
            llvm::errs() << "[MDMP PASS DEBUG] Successfully injected Wait instruction!\n";
            IRBuilder<> Builder(InsertPt);
            Builder.CreateCall(runtime_wait, {Req.RuntimeCall});
        }
    }
    PendingRequests.clear();
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "MDMP", "v0.2",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "mdmp") { MPM.addPass(MDMPPass()); return true; } return false;
                });
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel Level) { MPM.addPass(MDMPPass()); });
        }
    };
}
