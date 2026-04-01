#include "mdmp_compiler_pass.h"

using namespace llvm;

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

  if (!PendingRequests.empty()) {
    LLVMContext &Ctx = F.getContext();
    Module *M = F.getParent();
    //injectThrottledProgress(F, DT, LI, M);
    injectWaitsForRegion(nullptr, AA, LI, Ctx, M, DT);
  }
    
  return true;
}

void MDMPPass::transformFunctionsToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI) {
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();

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

  FunctionCallee runtime_send = M->getOrInsertFunction("mdmp_send", 
                                       Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                       Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
  if (Function *FSend = dyn_cast<Function>(runtime_send.getCallee())) { FSend->addFnAttr(Attribute::NoUnwind); }
 
  FunctionCallee runtime_recv = M->getOrInsertFunction("mdmp_recv", 
                                       Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                       Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
  if (Function *FRecv = dyn_cast<Function>(runtime_recv.getCallee())) { FRecv->addFnAttr(Attribute::NoUnwind); }

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

  FunctionCallee runtime_bcast = M->getOrInsertFunction("mdmp_bcast", 
                            Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                            Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx)); 
  if (Function *FBcast = dyn_cast<Function>(runtime_bcast.getCallee())) { FBcast->addFnAttr(Attribute::NoUnwind); }

  FunctionCallee runtime_register_send = M->getOrInsertFunction("mdmp_register_send", 
                                Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
  if (Function *FRegSend = dyn_cast<Function>(runtime_register_send.getCallee())) { FRegSend->addFnAttr(Attribute::NoUnwind); }
        
  FunctionCallee runtime_register_recv = M->getOrInsertFunction("mdmp_register_recv", 
                                Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
  if (Function *FRegRecv = dyn_cast<Function>(runtime_register_recv.getCallee())) { FRegRecv->addFnAttr(Attribute::NoUnwind); }

  FunctionCallee runtime_register_reduce = M->getOrInsertFunction("mdmp_register_reduce", 
                                  Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
                                  Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)); 
  if (Function *FRegReduce = dyn_cast<Function>(runtime_register_reduce.getCallee())) { FRegReduce->addFnAttr(Attribute::NoUnwind); }
        
  FunctionCallee runtime_register_gather = M->getOrInsertFunction("mdmp_register_gather", 
                                  Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                  PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx));
  if (Function *FRegGather = dyn_cast<Function>(runtime_register_gather.getCallee())) { FRegGather->addFnAttr(Attribute::NoUnwind); }

  FunctionCallee runtime_register_allreduce = M->getOrInsertFunction("mdmp_register_allreduce",
                                     Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), PointerType::getUnqual(Ctx), 
                                     Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx)); 
  if (Function *FRegAllreduce = dyn_cast<Function>(runtime_register_allreduce.getCallee())) { FRegAllreduce->addFnAttr(Attribute::NoUnwind); }

  FunctionCallee runtime_register_allgather = M->getOrInsertFunction("mdmp_register_allgather",
                                     Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                     PointerType::getUnqual(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)); 
  if (Function *FRegAllgather = dyn_cast<Function>(runtime_register_allgather.getCallee())) { FRegAllgather->addFnAttr(Attribute::NoUnwind); }
    
  FunctionCallee runtime_register_bcast = M->getOrInsertFunction("mdmp_register_bcast", 
                                 Type::getInt32Ty(Ctx), PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx), 
                                 Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx)); 
  if (Function *FRegBcast = dyn_cast<Function>(runtime_register_bcast.getCallee())) { FRegBcast->addFnAttr(Attribute::NoUnwind); }
  
  FunctionCallee runtime_commit = M->getOrInsertFunction("mdmp_commit", Type::getInt32Ty(Ctx));

  // Helper lambda to safely allocate state variables at the start of the function
  auto CreateSafeAlloc = [&]() -> AllocaInst* {
    BasicBlock &EntryBB = F.getEntryBlock();
    IRBuilder<> EntryBuilder(&EntryBB, EntryBB.begin());
    AllocaInst *Alloc = EntryBuilder.CreateAlloca(Type::getInt32Ty(Ctx), nullptr, "mdmp_req_id");
    EntryBuilder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), -1), Alloc); // Initialize to -1
    return Alloc;
  };

  std::vector<Instruction*> toDelete;
  std::vector<std::pair<AllocaInst*, std::vector<TrackedBuffer>>> ActiveDeclarativeLocs;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->getCalledFunction()) continue;
            
      StringRef Name = CI->getCalledFunction()->getName();
      IRBuilder<> Builder(CI);

      if (Name == "__mdmp_marker_commregion_begin") {
        CallInst *NewCall = Builder.CreateCall(runtime_begin);
        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.clear(); 
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_send" || Name == "__mdmp_marker_recv") {
        Value *BufferPtr = CI->getArgOperand(0);
        Value *CountVal  = CI->getArgOperand(1);
        Value *TypeVal   = CI->getArgOperand(2);
        Value *ByteSize  = CI->getArgOperand(3);
        Value *ActorRank = CI->getArgOperand(4);
        Value *PeerRank  = CI->getArgOperand(5);
        Value *TagVal    = CI->getArgOperand(6); 

        LocationSize LocSize = LocationSize::beforeOrAfterPointer();
        if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
        bool isSend = (Name == "__mdmp_marker_send");
        std::vector<TrackedBuffer> TrackedLocs = { {MemoryLocation(BufferPtr, LocSize), isSend} };
                
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        CallInst *NewCall = Builder.CreateCall(isSend ? runtime_send : runtime_recv, 
                                       {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});
        Builder.CreateStore(NewCall, ReqAlloc);
                
        CI->replaceAllUsesWith(NewCall);
        hoistInitiation(NewCall, TrackedLocs, AA, DT, LI, isSend);
        
        AsyncRequest Req;
        Req.WaitTokenAlloc = ReqAlloc;
        Req.StartPoint = NewCall;
        Req.Buffers = TrackedLocs;
        PendingRequests.push_back(Req);
        toDelete.push_back(CI);
      } 
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
        bool isSend = (Name == "__mdmp_marker_register_send");
        std::vector<TrackedBuffer> TrackedLocs = { {MemoryLocation(BufferPtr, LocSize), isSend} };
                
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        CallInst *NewCall = Builder.CreateCall(isSend ? runtime_register_send : runtime_register_recv, 
                                       {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});
        Builder.CreateStore(NewCall, ReqAlloc);
                
        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.push_back({ReqAlloc, TrackedLocs});
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_register_reduce") {
        Value *InBuf = CI->getArgOperand(0); Value *OutBuf = CI->getArgOperand(1);
        Value *ByteSize = CI->getArgOperand(4); 

        LocationSize LocSize = LocationSize::beforeOrAfterPointer();
        if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
                
        std::vector<TrackedBuffer> TrackedLocs = {
          {MemoryLocation(InBuf, LocSize), true}, {MemoryLocation(OutBuf, LocSize), false}
        };

        AllocaInst *ReqAlloc = CreateSafeAlloc();
        CallInst *NewCall = Builder.CreateCall(runtime_register_reduce, 
                                       {InBuf, OutBuf, CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
        Builder.CreateStore(NewCall, ReqAlloc);

        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.push_back({ReqAlloc, TrackedLocs});
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_register_gather") {
        Value *SendBuf = CI->getArgOperand(0); Value *RecvBuf = CI->getArgOperand(2);
        Value *ByteSize = CI->getArgOperand(4); 

        LocationSize LocSize = LocationSize::beforeOrAfterPointer();
        if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
                
        std::vector<TrackedBuffer> TrackedLocs = {
          {MemoryLocation(SendBuf, LocSize), true}, {MemoryLocation(RecvBuf, LocSize), false}
        };
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        CallInst *NewCall = Builder.CreateCall(runtime_register_gather, 
                                       {SendBuf, CI->getArgOperand(1), RecvBuf, CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
        Builder.CreateStore(NewCall, ReqAlloc);

        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.push_back({ReqAlloc, TrackedLocs});
        toDelete.push_back(CI);
      } 
      else if (Name == "__mdmp_marker_commit") {
        CallInst *NewCommit = Builder.CreateCall(runtime_commit);
        CI->replaceAllUsesWith(NewCommit);
                
        // Safely map the stack variables to the JIT Tracking array
        for (auto &Pair : ActiveDeclarativeLocs) {
          AsyncRequest Req;
          Req.WaitTokenAlloc = Pair.first;    
          Req.StartPoint = NewCommit;    
          Req.Buffers = Pair.second;
          PendingRequests.push_back(Req);
        }
                
        ActiveDeclarativeLocs.clear(); 
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_reduce") {
        CallInst *NewCall = Builder.CreateCall(runtime_reduce, 
                                       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
        
        std::vector<TrackedBuffer> Locs = {
          {MemoryLocation(CI->getArgOperand(1), LocationSize::beforeOrAfterPointer()), false}, 
          {MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer()), true}   
        };
        hoistInitiation(NewCall, Locs, AA, DT, LI, false);
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);

        CI->replaceAllUsesWith(NewCall);
        AsyncRequest Req;
        Req.WaitTokenAlloc = ReqAlloc;
        Req.StartPoint = NewCall;
        Req.Buffers = Locs;
        PendingRequests.push_back(Req);
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_gather") {
        CallInst *NewCall = Builder.CreateCall(runtime_gather, 
                                       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
        
        std::vector<TrackedBuffer> Locs = {
          {MemoryLocation(CI->getArgOperand(2), LocationSize::beforeOrAfterPointer()), false}, 
          {MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer()), true}   
        };
        hoistInitiation(NewCall, Locs, AA, DT, LI, false);
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);

        CI->replaceAllUsesWith(NewCall);
        AsyncRequest Req;
        Req.WaitTokenAlloc = ReqAlloc;
        Req.StartPoint = NewCall;
        Req.Buffers = Locs;
        PendingRequests.push_back(Req);
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_allreduce" || Name == "__mdmp_marker_register_allreduce") {
        FunctionCallee target_func = (Name == "__mdmp_marker_allreduce") ? runtime_allreduce : runtime_register_allreduce;
        CallInst *NewCall = Builder.CreateCall(target_func, 
                                       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
                
        std::vector<TrackedBuffer> Locs = {
          {MemoryLocation(CI->getArgOperand(1), LocationSize::beforeOrAfterPointer()), false}, 
          {MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer()), true}   
        };
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);
        CI->replaceAllUsesWith(NewCall);

        if (Name == "__mdmp_marker_allreduce") {
          hoistInitiation(NewCall, Locs, AA, DT, LI, false); 
          AsyncRequest Req;
          Req.WaitTokenAlloc = ReqAlloc;
          Req.StartPoint = NewCall;
          Req.Buffers = Locs;
          PendingRequests.push_back(Req);
        } else {
          ActiveDeclarativeLocs.push_back({ReqAlloc, Locs});
        }
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_allgather" || Name == "__mdmp_marker_register_allgather") {
        FunctionCallee target_func = (Name == "__mdmp_marker_allgather") ? runtime_allgather : runtime_register_allgather;
        CallInst *NewCall = Builder.CreateCall(target_func, 
                                       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});
                
        std::vector<TrackedBuffer> Locs = {
          {MemoryLocation(CI->getArgOperand(2), LocationSize::beforeOrAfterPointer()), false}, 
          {MemoryLocation(CI->getArgOperand(0), LocationSize::beforeOrAfterPointer()), true}  
        };

        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);
        CI->replaceAllUsesWith(NewCall);

        if (Name == "__mdmp_marker_allgather") {
          hoistInitiation(NewCall, Locs, AA, DT, LI, false);
          AsyncRequest Req;
          Req.WaitTokenAlloc = ReqAlloc;
          Req.StartPoint = NewCall;
          Req.Buffers = Locs;
          PendingRequests.push_back(Req);
        } else {
          ActiveDeclarativeLocs.push_back({ReqAlloc, Locs});
        }
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_bcast" || Name == "__mdmp_marker_register_bcast") {
        FunctionCallee target_func = (Name == "__mdmp_marker_bcast") ? runtime_bcast : runtime_register_bcast;
        CallInst *NewCall = Builder.CreateCall(target_func, 
                                       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});
        
        Value *ByteSize = CI->getArgOperand(3);
        LocationSize LocSize = LocationSize::beforeOrAfterPointer();
        if (auto *ConstBytes = dyn_cast<ConstantInt>(ByteSize)) { LocSize = LocationSize::precise(ConstBytes->getZExtValue()); }
        
        std::vector<TrackedBuffer> Locs = {
          {MemoryLocation(CI->getArgOperand(0), LocSize), false} 
        };
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);
        CI->replaceAllUsesWith(NewCall);

        if (Name == "__mdmp_marker_bcast") {
          hoistInitiation(NewCall, Locs, AA, DT, LI, false); 
          AsyncRequest Req;
          Req.WaitTokenAlloc = ReqAlloc;
          Req.StartPoint = NewCall;
          Req.Buffers = Locs;
          PendingRequests.push_back(Req);
        } else {
          ActiveDeclarativeLocs.push_back({ReqAlloc, Locs});
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

void MDMPPass::hoistInitiation(CallInst *CI, std::vector<TrackedBuffer> &Buffers, AAResults &AA, DominatorTree &DT, LoopInfo &LI, bool isSend) {
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
      for (auto &Buf : Buffers) {
        if (auto *SI = dyn_cast<StoreInst>(Prev)) {
          AliasResult AR = AA.alias(MemoryLocation::get(SI), Buf.Loc);
          bool underlyingMatch = (getUnderlyingObject(SI->getPointerOperand()) == getUnderlyingObject(Buf.Loc.Ptr));
          if (AR == AliasResult::MustAlias || AR == AliasResult::MayAlias || underlyingMatch) { 
              memoryCollision = true; break; 
          }
        } 
        else if (auto *LInst = dyn_cast<LoadInst>(Prev)) {
          if (!Buf.isNetworkReadOnly) { 
            AliasResult AR = AA.alias(MemoryLocation::get(LInst), Buf.Loc);
            bool underlyingMatch = (getUnderlyingObject(LInst->getPointerOperand()) == getUnderlyingObject(Buf.Loc.Ptr));
            if (AR == AliasResult::MustAlias || AR == AliasResult::MayAlias || underlyingMatch) { 
                memoryCollision = true; break; 
            }
          }
        } 
        else {
          auto MR = AA.getModRefInfo(Prev, Buf.Loc);
          if (Buf.isNetworkReadOnly && isModSet(MR)) { memoryCollision = true; break; }
          else if (!Buf.isNetworkReadOnly && isModOrRefSet(MR)) { memoryCollision = true; break; }
        }
      }
    }
        
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
        
    BasicBlock::iterator StartIt = Req.StartPoint->getIterator();
    StartIt++;
    Worklist.push_back({Req.StartPoint->getParent(), StartIt});
        
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
                  FnName == "mdmp_gather" || FnName == "mdmp_bcast" ||
                  FnName == "mdmp_commit" || 
                  FnName == "mdmp_wait" || FnName == "mdmp_wtime" || 
                  FnName == "mdmp_register_send" || FnName == "mdmp_register_recv" ||
                  FnName == "mdmp_register_reduce" || FnName == "mdmp_register_gather" ||
                  FnName == "mdmp_register_allreduce" || FnName == "mdmp_register_allgather" ||
                  FnName == "mdmp_register_bcast" ||
                  FnName.starts_with("__mdmp_marker_")) {
                isAsyncCall = true;
              }
            }
          }
                    
          bool memoryCollision = false;

          if (auto *Call = dyn_cast<CallInst>(Inst)) {
            if (Function *CalledFn = Call->getCalledFunction()) {
              StringRef Name = CalledFn->getName();
              bool isMDMPBarrier = (Name == "mdmp_final" || Name == "__mdmp_marker_final" || 
                                    Name == "mdmp_abort" || Name == "__mdmp_marker_abort" ||
                                    Name == "mdmp_sync"  || Name == "__mdmp_marker_sync");
                                        
              bool isMPIBarrier = Name.starts_with("MPI_Wait") || Name == "MPI_Barrier" ||
                Name == "MPI_Bcast" || Name == "MPI_Allreduce" || Name == "MPI_Reduce" || 
                Name == "MPI_Gather" || Name == "MPI_Scatter" || Name == "MPI_Alltoall" || 
                Name == "MPI_Allgather" || Name == "MPI_Finalize";
                                  
              bool isMPIP2P = (Name == "MPI_Send" || Name == "MPI_Recv" || Name == "MPI_Sendrecv");
              bool isMPITopology = Name.contains("MPI_Comm_") || Name == "MPI_Cart_create";
              
              if (Name == "MPI_Comm_rank" || Name == "MPI_Comm_size") isMPITopology = false;

              if (isMDMPBarrier || isMPIBarrier || isMPIP2P || isMPITopology) memoryCollision = true;
            }
          }

          if (!memoryCollision && !isAsyncCall) {
            for (auto &Buf : Req.Buffers) {
              if (auto *LInst = dyn_cast<LoadInst>(Inst)) {
                if (!Buf.isNetworkReadOnly) { 
                  AliasResult AR = AA.alias(MemoryLocation::get(LInst), Buf.Loc);
                  bool underlyingMatch = (getUnderlyingObject(LInst->getPointerOperand()) == getUnderlyingObject(Buf.Loc.Ptr));
                  if (AR == AliasResult::MustAlias || AR == AliasResult::MayAlias || underlyingMatch) { memoryCollision = true; break; }
                }
              } 
              else if (auto *SInst = dyn_cast<StoreInst>(Inst)) {
                AliasResult AR = AA.alias(MemoryLocation::get(SInst), Buf.Loc);
                bool underlyingMatch = (getUnderlyingObject(SInst->getPointerOperand()) == getUnderlyingObject(Buf.Loc.Ptr));
                if (AR == AliasResult::MustAlias || AR == AliasResult::MayAlias || underlyingMatch) { memoryCollision = true; break; }
              } 
              else if (Inst->mayReadOrWriteMemory()) {
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
        Loop *ReqLoop = LI.getLoopFor(Req.StartPoint->getParent());

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
            if (!ReqLoop || EdgeLoop == ReqLoop || EdgeLoop->contains(ReqLoop)) {
              WaitInsertionPoints.push_back(BB->getTerminator());
              continue; 
            }
          }
          Worklist.push_back({Succ, Succ->begin()});
        }
                
        if (!hasSuccessors) {
          WaitInsertionPoints.push_back(BB->getTerminator());
        }
      } 
    }
        
    SmallPtrSet<Instruction*, 4> UniqueWaitPoints;
    for (Instruction *InsertPt : WaitInsertionPoints) {
            
      if (!DT.dominates(Req.StartPoint, InsertPt)) {
        InsertPt = Req.StartPoint->getParent()->getTerminator();
      }

      Instruction *HoistPt = InsertPt;
      Loop *L = LI.getLoopFor(HoistPt->getParent());
      Loop *ReqLoop = LI.getLoopFor(Req.StartPoint->getParent());

      while (L && L != ReqLoop) {
        BasicBlock *Latch = L->getLoopLatch();
        if (Latch && DT.dominates(HoistPt->getParent(), Latch)) {
          BasicBlock *Preheader = L->getLoopPreheader();
          if (Preheader) {
            Instruction *PotentialHoistPt = Preheader->getTerminator();
            if (DT.dominates(Req.StartPoint, PotentialHoistPt)) {
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
      IRBuilder<> Builder(InsertPt);
      // Safely load the ID from the stack variable before passing it to mdmp_wait
      LoadInst *LoadedID = Builder.CreateLoad(Type::getInt32Ty(Ctx), Req.WaitTokenAlloc);
      Builder.CreateCall(runtime_wait, {LoadedID}); 
    }
  }
  PendingRequests.clear();
}

void MDMPPass::injectThrottledProgress(Function &F, DominatorTree &DT, LoopInfo &LI, Module *M) {
  LLVMContext &Ctx = M->getContext();
  FunctionCallee runtime_progress = M->getOrInsertFunction("mdmp_progress", Type::getVoidTy(Ctx));

  BasicBlock &EntryBB = F.getEntryBlock();
  IRBuilder<> EntryBuilder(&EntryBB, EntryBB.begin());
  
  AllocaInst *CounterAlloc = EntryBuilder.CreateAlloca(Type::getInt32Ty(Ctx), nullptr, "tickle_counter");
  EntryBuilder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0), CounterAlloc);

  for (Loop *L : LI) {
    if (!L->getSubLoops().empty()) continue; 

    BasicBlock *Header = L->getHeader();

    bool isInFlight = false;
    for (auto &Req : PendingRequests) {
      if (DT.dominates(Req.StartPoint->getParent(), Header)) {
        isInFlight = true;
        break;
      }
    }

    if (!isInFlight) continue;

    Instruction *InsertPt = &*Header->getFirstInsertionPt();
    IRBuilder<> LoopBuilder(InsertPt);

    LoadInst *CurrentCount = LoopBuilder.CreateLoad(Type::getInt32Ty(Ctx), CounterAlloc);
    Value *NextCount = LoopBuilder.CreateAdd(CurrentCount, ConstantInt::get(Type::getInt32Ty(Ctx), 1));
    LoopBuilder.CreateStore(NextCount, CounterAlloc);

    Value *ModResult = LoopBuilder.CreateURem(NextCount, ConstantInt::get(Type::getInt32Ty(Ctx), 64));
    Value *IsZero = LoopBuilder.CreateICmpEQ(ModResult, ConstantInt::get(Type::getInt32Ty(Ctx), 0));

    Instruction *SplitTerminator = SplitBlockAndInsertIfThen(IsZero, InsertPt, false);
    
    IRBuilder<> ThenBuilder(SplitTerminator);
    ThenBuilder.CreateCall(runtime_progress);
  }
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "MDMP", "v0.2",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback([](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
           if (Name == "mdmp") { MPM.addPass(MDMPPass()); return true; } return false;
      });
      PB.registerOptimizerLastEPCallback([](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) { 
           MPM.addPass(MDMPPass()); 
      });
      PB.registerFullLinkTimeOptimizationLastEPCallback([](ModulePassManager &MPM, OptimizationLevel Level) { 
           MPM.addPass(MDMPPass()); 
      });
    }
  };
}
