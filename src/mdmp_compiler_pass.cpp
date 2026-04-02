#include "mdmp_compiler_pass.h"

using namespace llvm;

bool MDMPPass::waitTokenValueDominates(Value *V, Instruction *InsertPt, DominatorTree &DT) {
                                    
  if (!V) return false;

  if (isa<Constant>(V) || isa<Argument>(V) || isa<GlobalValue>(V))
    return true;

  if (auto *I = dyn_cast<Instruction>(V))
    return DT.dominates(I, InsertPt);

  return true;
}

Value *MDMPPass::materialiseWaitTokenForUse(AsyncRequest &Req, Instruction *InsertPt, IntegerType *I32Ty, IRBuilder<> &Builder, DominatorTree &DT) {

  if (Req.WaitTokenValue &&
      waitTokenValueDominates(Req.WaitTokenValue, InsertPt, DT)) {
    return Req.WaitTokenValue;
  }

  assert(Req.WaitTokenAlloc &&
         "MDMP request has no available wait token at insertion point");

  return Builder.CreateLoad(I32Ty, Req.WaitTokenAlloc, "mdmp_wait_id");
}


bool MDMPPass::isAsyncMDMPOpName(StringRef FnName) {
  return FnName == "mdmp_send" ||
    FnName == "mdmp_recv" ||
    FnName == "mdmp_reduce" ||
    FnName == "mdmp_allreduce" ||
    FnName == "mdmp_allgather" ||
    FnName == "mdmp_gather" ||
    FnName == "mdmp_bcast" ||
    FnName == "mdmp_commit" ||
    FnName == "mdmp_register_send" ||
    FnName == "mdmp_register_recv" ||
    FnName == "mdmp_register_reduce" ||
    FnName == "mdmp_register_gather" ||
    FnName == "mdmp_register_allreduce" ||
    FnName == "mdmp_register_allgather" ||
    FnName == "mdmp_register_bcast" ||
    FnName.starts_with("__mdmp_marker_");
}

bool MDMPPass::isHardBarrierCallName(StringRef Name) {
  bool isMDMPBarrier =
    (Name == "mdmp_final" || Name == "__mdmp_marker_final" ||
     Name == "mdmp_abort" || Name == "__mdmp_marker_abort" ||
     Name == "mdmp_sync"  || Name == "__mdmp_marker_sync"  ||
     Name == "mdmp_wait"  ||
     Name == "mdmp_wait_many" ||
     Name == "mdmp_commregion_end" ||
     Name == "__mdmp_marker_commregion_end" ||
     // Keep this conservative until multi-batch declarative commits are fully supported.
     Name == "mdmp_commit" || Name == "__mdmp_marker_commit");

  bool isMPIBarrier =
    Name.starts_with("MPI_Wait") ||
    Name == "MPI_Barrier" ||
    Name == "MPI_Bcast" ||
    Name == "MPI_Allreduce" ||
    Name == "MPI_Reduce" ||
    Name == "MPI_Gather" ||
    Name == "MPI_Scatter" ||
    Name == "MPI_Alltoall" ||
    Name == "MPI_Allgather" ||
    Name == "MPI_Finalize";

  bool isMPIP2P =
    (Name == "MPI_Send" || Name == "MPI_Recv" || Name == "MPI_Sendrecv");

  bool isMPITopology =
    Name.contains("MPI_Comm_") || Name == "MPI_Cart_create";

  if (Name == "MPI_Comm_rank" || Name == "MPI_Comm_size")
    isMPITopology = false;

  return isMDMPBarrier || isMPIBarrier || isMPIP2P || isMPITopology;
}

void MDMPPass::collectLeafLoops(Loop *L, SmallVectorImpl<Loop *> &Out) {
  if (L->getSubLoops().empty()) {
    Out.push_back(L);
    return;
  }

  for (Loop *SubL : L->getSubLoops()) {
    collectLeafLoops(SubL, Out);
  }
}

void MDMPPass::collectLeafLoops(LoopInfo &LI, SmallVectorImpl<Loop *> &Out) {
  for (Loop *TopL : LI) {
    collectLeafLoops(TopL, Out);
  }
}

bool MDMPPass::requestWindowCoversLoopHeader(const RequestWindowInfo &Info, Loop *L, DominatorTree &DT) {
                                          
                                          
  BasicBlock *Header = L->getHeader();
  if (!Info.LiveBlocks.contains(Header))
    return false;

  Instruction *HeaderIP = &*Header->getFirstInsertionPt();
  return DT.dominates(Info.Req->StartPoint, HeaderIP);
}

SmallVector<MDMPPass::RequestWindowInfo, 8> MDMPPass::analyseRequestWindows(std::vector<AsyncRequest> &Requests, Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, Module *M) {                      

  const DataLayout &DL = M->getDataLayout();
  SmallVector<RequestWindowInfo, 8> Infos;
  Infos.reserve(Requests.size());

  for (AsyncRequest &Req : Requests) {
    RequestWindowInfo Info;
    Info.Req = &Req;

    SmallPtrSet<BasicBlock *, 16> VisitedFromBegin;
    SmallPtrSet<BasicBlock *, 16> VisitedPartial;
    SmallVector<TraversalState, 16> Worklist;

    auto StartIt = Req.StartPoint->getIterator();
    ++StartIt;
    Worklist.push_back({Req.StartPoint->getParent(), StartIt});

    while (!Worklist.empty()) {
      TraversalState State = Worklist.pop_back_val();
      BasicBlock *BB = State.BB;

      bool StartsAtBegin = (State.StartIt == BB->begin());
      if (StartsAtBegin) {
        if (!VisitedFromBegin.insert(BB).second)
          continue;
      } else {
        if (!VisitedPartial.insert(BB).second)
          continue;
      }

      Info.LiveBlocks.insert(BB);

      bool FoundStop = false;
      for (auto It = State.StartIt; It != BB->end(); ++It) {
        Instruction *Inst = &*It;

        if (RegionEnd && Inst == RegionEnd) {
          Info.WaitPoints.push_back(Inst);
          FoundStop = true;
          break;
        }

        if (!Inst->mayReadOrWriteMemory())
          continue;

        bool StopHere = false;
        bool IsAsyncCall = false;

        if (auto *CB = dyn_cast<CallBase>(Inst)) {
          if (Function *CalledFn = CB->getCalledFunction()) {
            StringRef Name = CalledFn->getName();
            IsAsyncCall = isAsyncMDMPOpName(Name);

            if (isHardBarrierCallName(Name))
              StopHere = true;
          }
        }

        if (!StopHere && !IsAsyncCall) {
          StopHere =
	    instructionConflictsWithAnyTrackedBuffer(Inst, Req.Buffers, AA, DL);
        }

        if (StopHere) {
          Info.WaitPoints.push_back(Inst);
          FoundStop = true;
          break;
        }
      }

      if (FoundStop)
        continue;

      bool HasSuccessors = false;
      Loop *ReqLoop = LI.getLoopFor(Req.StartPoint->getParent());

      for (BasicBlock *Succ : successors(BB)) {
        HasSuccessors = true;

        bool IsBackedge = false;
        Loop *EdgeLoop = LI.getLoopFor(BB);
        while (EdgeLoop) {
          if (EdgeLoop->getHeader() == Succ) {
            IsBackedge = true;
            break;
          }
          EdgeLoop = EdgeLoop->getParentLoop();
        }

        if (IsBackedge && EdgeLoop) {
          if (!ReqLoop || EdgeLoop == ReqLoop || EdgeLoop->contains(ReqLoop)) {
            Info.WaitPoints.push_back(BB->getTerminator());
            continue;
          }
        }

        Worklist.push_back({Succ, Succ->begin()});
      }

      if (!HasSuccessors) {
        Info.WaitPoints.push_back(BB->getTerminator());
      }
    }

    Infos.push_back(std::move(Info));
  }

  return Infos;
}

std::optional<uint64_t> MDMPPass::getConstU64(Value *V) {
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->getZExtValue();
  return std::nullopt;
}

std::optional<uint64_t> MDMPPass::getStaticMPITypeBytes(Value *TypeCodeV) {
  auto TC = getConstU64(TypeCodeV);
  if (!TC) return std::nullopt;

  switch (*TC) {
  case 0: return 4; // int
  case 1: return 8; // double
  case 2: return 4; // float
  case 3: return 1; // char
  case 4: return 1; // byte
  default: return std::nullopt;
  }
}

std::optional<uint64_t> MDMPPass::checkedMulU64(uint64_t A, uint64_t B) {
  if (A != 0 && B > UINT64_MAX / A) return std::nullopt;
  return A * B;
}

LocationSize MDMPPass::derivePreciseSpan(Value *CountV, Value *TypeCodeV, Value *BytesV) {
  if (auto Bytes = getConstU64(BytesV))
    return LocationSize::precise(*Bytes);

  auto Count = getConstU64(CountV);
  auto TypeBytes = getStaticMPITypeBytes(TypeCodeV);
  if (Count && TypeBytes) {
    if (auto Total = checkedMulU64(*Count, *TypeBytes))
      return LocationSize::precise(*Total);
  }

  return LocationSize::beforeOrAfterPointer();
}

MDMPPass::TrackedBuffer MDMPPass::makePreciseTrackedBuffer(Value *Ptr, Value *CountV, Value *TypeCodeV,
							   Value *BytesV, bool IsNetworkReadOnly) {
  return { MemoryLocation(Ptr, derivePreciseSpan(CountV, TypeCodeV, BytesV)),
    IsNetworkReadOnly };
}

MDMPPass::TrackedBuffer MDMPPass::makeUnknownTrackedBuffer(Value *Ptr, bool IsNetworkReadOnly) {
  return { MemoryLocation(Ptr, LocationSize::beforeOrAfterPointer()),
    IsNetworkReadOnly };
}

// Note: if your LLVM version's LocationSize::getValue() returns TypeSize,
// replace "return S.getValue();" with:
//   return S.getValue().getKnownMinValue();
std::optional<uint64_t> MDMPPass::getPreciseSizeBytes(LocationSize S) {
  if (!S.hasValue() || !S.isPrecise())
    return std::nullopt;
  return S.getValue();
}

bool MDMPPass::areDefinitelyDisjoint(const MemoryLocation &A,
				     const MemoryLocation &B,
				     const DataLayout &DL) {
  int64_t OffA = 0, OffB = 0;
  const Value *BaseA = GetPointerBaseWithConstantOffset(A.Ptr, OffA, DL);
  const Value *BaseB = GetPointerBaseWithConstantOffset(B.Ptr, OffB, DL);

  if (!BaseA || !BaseB)
    return false;

  const Value *UA = getUnderlyingObject(BaseA);
  const Value *UB = getUnderlyingObject(BaseB);

  // Different identified objects => cannot alias.
  if (UA != UB && isIdentifiedObject(UA) && isIdentifiedObject(UB))
    return true;

  // Same underlying object with exact byte intervals.
  if (UA == UB) {
    auto SA = getPreciseSizeBytes(A.Size);
    auto SB = getPreciseSizeBytes(B.Size);
    if (!SA || !SB)
      return false;

    int64_t AStart = OffA;
    int64_t AEnd   = OffA + static_cast<int64_t>(*SA);
    int64_t BStart = OffB;
    int64_t BEnd   = OffB + static_cast<int64_t>(*SB);

    return (AEnd <= BStart) || (BEnd <= AStart);
  }

  return false;
}

bool MDMPPass::locationsMayOverlap(const MemoryLocation &A,
				   const MemoryLocation &B,
				   AAResults &AA,
				   const DataLayout &DL) {
  if (areDefinitelyDisjoint(A, B, DL))
    return false;

  return AA.alias(A, B) != AliasResult::NoAlias;
}

bool MDMPPass::isHardMotionBarrier(Instruction *I) {
  if (isa<FenceInst>(I) || isa<AtomicCmpXchgInst>(I) || isa<AtomicRMWInst>(I))
    return true;

  if (auto *LI = dyn_cast<LoadInst>(I))
    if (!LI->isSimple()) return true;

  if (auto *SI = dyn_cast<StoreInst>(I))
    if (!SI->isSimple()) return true;

  if (auto *CB = dyn_cast<CallBase>(I)) {
    if (CB->isInlineAsm())
      return true;

    if (CB->mayThrow())
      return true;

    if (Function *Callee = CB->getCalledFunction()) {
      StringRef N = Callee->getName();
      if (N == "mdmp_commregion_begin" ||
          N.starts_with("mdmp_") ||
          N.starts_with("MPI_") ||
          N.starts_with("__mdmp_marker_")) {
        return true;
      }
    }

    // Pure/readnone/readonly calls can still be moved across if operands dominate
    // and there is no memory conflict with tracked buffers.
    if (CB->mayHaveSideEffects() &&
        !CB->doesNotAccessMemory() &&
        !CB->onlyReadsMemory()) {
      return true;
    }
  }

  return false;
}

bool MDMPPass::operandsAvailableBefore(CallInst *CI,
				       Instruction *InsertBefore,
				       DominatorTree &DT) {
  for (Use &U : CI->args()) {
    if (auto *OpI = dyn_cast<Instruction>(U.get())) {
      if (OpI == InsertBefore)
        return false;
      if (!DT.dominates(OpI, InsertBefore))
        return false;
    }
  }
  return true;
}

bool MDMPPass::instructionConflictsWithTrackedBuffer(Instruction *I,
						     const TrackedBuffer &Buf,
						     AAResults &AA,
						     const DataLayout &DL) {
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    // Local reads are okay while a send is in flight.
    if (Buf.isNetworkReadOnly)
      return false;
    return locationsMayOverlap(MemoryLocation::get(LI), Buf.Loc, AA, DL);
  }

  if (auto *SI = dyn_cast<StoreInst>(I)) {
    // Local writes are never okay if they overlap the network buffer.
    return locationsMayOverlap(MemoryLocation::get(SI), Buf.Loc, AA, DL);
  }

  auto MR = AA.getModRefInfo(I, Buf.Loc);
  if (Buf.isNetworkReadOnly)
    return isModSet(MR);
  return isModOrRefSet(MR);
}

bool MDMPPass::instructionConflictsWithAnyTrackedBuffer(Instruction *I,
							ArrayRef<TrackedBuffer> Buffers,
							AAResults &AA,
							const DataLayout &DL) {
  for (const TrackedBuffer &Buf : Buffers) {
    if (instructionConflictsWithTrackedBuffer(I, Buf, AA, DL))
      return true;
  }
  return false;
}

BasicBlock *MDMPPass::getLinearPredecessor(BasicBlock *BB) {
  BasicBlock *Pred = BB->getSinglePredecessor();
  if (!Pred) return nullptr;
  if (Pred->getSingleSuccessor() != BB) return nullptr;
  return Pred;
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildSendRecvBuffers(Value *Buf, Value *Count,
								    Value *Type, Value *Bytes,
								    bool IsSend) {
  return { makePreciseTrackedBuffer(Buf, Count, Type, Bytes, IsSend) };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildReduceBuffers(Value *SendBuf, Value *RecvBuf,
								  Value *Count, Value *Type,
								  Value *Bytes) {
  return {
    makePreciseTrackedBuffer(RecvBuf, Count, Type, Bytes, false),
    makePreciseTrackedBuffer(SendBuf, Count, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildGatherBuffers(Value *SendBuf, Value *SendCount,
								  Value *RecvBuf, Value *Type,
								  Value *Bytes) {
  // Send side is exact. Receive side depends on root/global_size, so keep
  // conservative unless your frontend can pass total recv-buffer bytes.
  return {
    makeUnknownTrackedBuffer(RecvBuf, false),
    makePreciseTrackedBuffer(SendBuf, SendCount, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildAllreduceBuffers(Value *SendBuf, Value *RecvBuf,
								     Value *Count, Value *Type,
								     Value *Bytes) {
  return {
    makePreciseTrackedBuffer(RecvBuf, Count, Type, Bytes, false),
    makePreciseTrackedBuffer(SendBuf, Count, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildAllgatherBuffers(Value *SendBuf, Value *Count,
								     Value *RecvBuf, Value *Type,
								     Value *Bytes) {
  return {
    makeUnknownTrackedBuffer(RecvBuf, false),
    makePreciseTrackedBuffer(SendBuf, Count, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildBcastBuffers(Value *Buf, Value *Count,
								 Value *Type, Value *Bytes) {
  // Conservatively treat as recv-like because non-root ranks get writes.
  return {
    makePreciseTrackedBuffer(Buf, Count, Type, Bytes, false)
  };
}


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

    auto SavedRequests = PendingRequests;

    injectWaitsForRegion(nullptr, AA, LI, Ctx, M, DT);

    PendingRequests = std::move(SavedRequests);
    injectThrottledProgress(F, AA, DT, LI, M);
 
    PendingRequests.clear();
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
  struct DeclarativePendingOp {
    Value *WaitTokenValue;
    AllocaInst *WaitTokenAlloc;
    std::vector<TrackedBuffer> Buffers;
  };
  
  std::vector<DeclarativePendingOp> ActiveDeclarativeLocs;
 
  
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

        bool isSend = (Name == "__mdmp_marker_send");
        std::vector<TrackedBuffer> TrackedLocs = buildSendRecvBuffers(BufferPtr, CountVal, TypeVal, ByteSize, isSend);
	CallInst *NewCall = Builder.CreateCall(isSend ? runtime_send : runtime_recv,
					       {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});

	CI->replaceAllUsesWith(NewCall);
	hoistInitiation(NewCall, TrackedLocs, AA, DT, LI, isSend);
	
	AsyncRequest Req;
	Req.WaitTokenValue = NewCall;
	Req.StartPoint = NewCall;
	Req.Buffers = TrackedLocs;
	PendingRequests.push_back(std::move(Req));
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

        bool isSend = (Name == "__mdmp_marker_register_send");
        std::vector<TrackedBuffer> TrackedLocs = buildSendRecvBuffers(BufferPtr, CountVal, TypeVal, ByteSize, isSend);

	AllocaInst *ReqAlloc = CreateSafeAlloc();
	CallInst *NewCall = Builder.CreateCall(isSend ? runtime_register_send : runtime_register_recv,
					       {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});
	Builder.CreateStore(NewCall, ReqAlloc);

	CI->replaceAllUsesWith(NewCall);
	ActiveDeclarativeLocs.push_back({NewCall, ReqAlloc, TrackedLocs});
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_register_reduce") {
        Value *InBuf = CI->getArgOperand(0); Value *OutBuf = CI->getArgOperand(1);
        Value *ByteSize = CI->getArgOperand(4); 

	std::vector<TrackedBuffer> TrackedLocs = buildReduceBuffers(InBuf, OutBuf,
								    CI->getArgOperand(2),  // count
								    CI->getArgOperand(3),  // type
								    CI->getArgOperand(4)); // bytes

	AllocaInst *ReqAlloc = CreateSafeAlloc();
	CallInst *NewCall = Builder.CreateCall(runtime_register_reduce,
					       {InBuf, OutBuf, CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
	Builder.CreateStore(NewCall, ReqAlloc);
	
	CI->replaceAllUsesWith(NewCall);
	ActiveDeclarativeLocs.push_back({NewCall, ReqAlloc, TrackedLocs});
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_register_gather") {
        Value *SendBuf = CI->getArgOperand(0); Value *RecvBuf = CI->getArgOperand(2);
        Value *ByteSize = CI->getArgOperand(4); 

        std::vector<TrackedBuffer> TrackedLocs = buildGatherBuffers(SendBuf,
                                                                    CI->getArgOperand(1),  // sendcount
                                                                    RecvBuf,
                                                                    CI->getArgOperand(3),  // type
                                                                    CI->getArgOperand(4)); // bytes
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
	CallInst *NewCall = Builder.CreateCall(runtime_register_gather,
					       {SendBuf, CI->getArgOperand(1), RecvBuf, CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
	Builder.CreateStore(NewCall, ReqAlloc);
	
	CI->replaceAllUsesWith(NewCall);
	ActiveDeclarativeLocs.push_back({NewCall, ReqAlloc, TrackedLocs});
	toDelete.push_back(CI);
      } 
      else if (Name == "__mdmp_marker_commit") {
        CallInst *NewCommit = Builder.CreateCall(runtime_commit);
        CI->replaceAllUsesWith(NewCommit);
                
        // Safely map the stack variables to the JIT Tracking array
	for (auto &Op : ActiveDeclarativeLocs) {
	  AsyncRequest Req;
	  Req.WaitTokenValue = Op.WaitTokenValue;
	  Req.WaitTokenAlloc = Op.WaitTokenAlloc;
	  Req.StartPoint = NewCommit;
	  Req.Buffers = Op.Buffers;
	  PendingRequests.push_back(std::move(Req));
	}

                
        ActiveDeclarativeLocs.clear(); 
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_reduce") {
        CallInst *NewCall = Builder.CreateCall(runtime_reduce, 
					       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
       
        std::vector<TrackedBuffer> Locs =
	  buildReduceBuffers(CI->getArgOperand(0),  // sendbuf
			     CI->getArgOperand(1),  // recvbuf
			     CI->getArgOperand(2),  // count
			     CI->getArgOperand(3),  // type
			     CI->getArgOperand(4)); // bytes 

	hoistInitiation(NewCall, Locs, AA, DT, LI, false);

	CI->replaceAllUsesWith(NewCall);
	AsyncRequest Req;
	Req.WaitTokenValue = NewCall;
	Req.StartPoint = NewCall;
	Req.Buffers = Locs;
	PendingRequests.push_back(std::move(Req));

        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_gather") {
        CallInst *NewCall = Builder.CreateCall(runtime_gather, 
					       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
        
        std::vector<TrackedBuffer> Locs =
	  buildGatherBuffers(CI->getArgOperand(0),  // sendbuf
			     CI->getArgOperand(1),  // sendcount
			     CI->getArgOperand(2),  // recvbuf
			     CI->getArgOperand(3),  // type
			     CI->getArgOperand(4)); // bytes
	hoistInitiation(NewCall, Locs, AA, DT, LI, false);

	CI->replaceAllUsesWith(NewCall);
	AsyncRequest Req;
	Req.WaitTokenValue = NewCall;
	Req.StartPoint = NewCall;
	Req.Buffers = Locs;
	PendingRequests.push_back(std::move(Req));

        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_allreduce" || Name == "__mdmp_marker_register_allreduce") {
        FunctionCallee target_func = (Name == "__mdmp_marker_allreduce") ? runtime_allreduce : runtime_register_allreduce;
        CallInst *NewCall = Builder.CreateCall(target_func, 
					       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
                
        std::vector<TrackedBuffer> Locs =
	  buildAllreduceBuffers(CI->getArgOperand(0),  // sendbuf
				CI->getArgOperand(1),  // recvbuf
				CI->getArgOperand(2),  // count
				CI->getArgOperand(3),  // type
				CI->getArgOperand(4)); // bytes
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);

	CI->replaceAllUsesWith(NewCall);

	if (Name == "__mdmp_marker_allreduce") {
	  hoistInitiation(NewCall, Locs, AA, DT, LI, false);
	  AsyncRequest Req;
	  Req.WaitTokenValue = NewCall;
	  Req.StartPoint = NewCall;
	  Req.Buffers = Locs;
	  PendingRequests.push_back(std::move(Req));
	} else {
	  AllocaInst *ReqAlloc = CreateSafeAlloc();
	  Builder.CreateStore(NewCall, ReqAlloc);
	  ActiveDeclarativeLocs.push_back({NewCall, ReqAlloc, Locs});
	}
	toDelete.push_back(CI);

      }
      else if (Name == "__mdmp_marker_allgather" || Name == "__mdmp_marker_register_allgather") {
        FunctionCallee target_func = (Name == "__mdmp_marker_allgather") ? runtime_allgather : runtime_register_allgather;
        CallInst *NewCall = Builder.CreateCall(target_func, 
					       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});
                
        std::vector<TrackedBuffer> Locs =
	  buildAllgatherBuffers(CI->getArgOperand(0),  // sendbuf
				CI->getArgOperand(1),  // count
				CI->getArgOperand(2),  // recvbuf
				CI->getArgOperand(3),  // type
				CI->getArgOperand(4)); // bytes

        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);
	CI->replaceAllUsesWith(NewCall);

	if (Name == "__mdmp_marker_allgather") {
	  hoistInitiation(NewCall, Locs, AA, DT, LI, false);
	  AsyncRequest Req;
	  Req.WaitTokenValue = NewCall;
	  Req.StartPoint = NewCall;
	  Req.Buffers = Locs;
	  PendingRequests.push_back(std::move(Req));
	} else {
	  AllocaInst *ReqAlloc = CreateSafeAlloc();
	  Builder.CreateStore(NewCall, ReqAlloc);
	  ActiveDeclarativeLocs.push_back({NewCall, ReqAlloc, Locs});
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
        
        std::vector<TrackedBuffer> Locs =
	  buildBcastBuffers(CI->getArgOperand(0),  // buffer
			    CI->getArgOperand(1),  // count
			    CI->getArgOperand(2),  // type
			    CI->getArgOperand(3)); // bytes
        
        AllocaInst *ReqAlloc = CreateSafeAlloc();
        Builder.CreateStore(NewCall, ReqAlloc);

	CI->replaceAllUsesWith(NewCall);

	if (Name == "__mdmp_marker_bcast") {
	  hoistInitiation(NewCall, Locs, AA, DT, LI, false);
	  AsyncRequest Req;
	  Req.WaitTokenValue = NewCall;
	  Req.StartPoint = NewCall;
	  Req.Buffers = Locs;
	  PendingRequests.push_back(std::move(Req));
	} else {
	  AllocaInst *ReqAlloc = CreateSafeAlloc();
	  Builder.CreateStore(NewCall, ReqAlloc);
	  ActiveDeclarativeLocs.push_back({NewCall, ReqAlloc, Locs});
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
  (void)LI;
  (void)isSend;

  const DataLayout &DL = CI->getModule()->getDataLayout();

  Instruction *InsertBefore = CI;
  BasicBlock *CurBB = CI->getParent();

  while (true) {
    bool ReachedBlockFront = true;

    while (Instruction *Prev = InsertBefore->getPrevNode()) {
      if (isa<PHINode>(Prev))
        break;

      if (!operandsAvailableBefore(CI, Prev, DT)) {
        ReachedBlockFront = false;
        break;
      }

      if (isHardMotionBarrier(Prev)) {
        ReachedBlockFront = false;
        break;
      }

      if (Prev->mayReadOrWriteMemory() &&
          instructionConflictsWithAnyTrackedBuffer(Prev, Buffers, AA, DL)) {
        ReachedBlockFront = false;
        break;
      }

      InsertBefore = Prev;
    }

    if (!ReachedBlockFront)
      break;

    BasicBlock *Pred = getLinearPredecessor(CurBB);
    if (!Pred)
      break;

    Instruction *PredTerm = Pred->getTerminator();
    if (!operandsAvailableBefore(CI, PredTerm, DT))
      break;

    // Move into the predecessor and continue scanning there.
    InsertBefore = PredTerm;
    CurBB = Pred;
  }

  if (InsertBefore != CI)
    CI->moveBefore(InsertBefore->getIterator());
}

void MDMPPass::injectWaitsForRegion(Instruction *RegionEnd,
                                    AAResults &AA,
                                    LoopInfo &LI,
                                    LLVMContext &Ctx,
                                    Module *M,
                                    DominatorTree &DT) {
  IntegerType *I32Ty = Type::getInt32Ty(Ctx);

  FunctionCallee runtime_wait =
    M->getOrInsertFunction("mdmp_wait", Type::getVoidTy(Ctx), I32Ty);

  FunctionCallee runtime_wait_many =
    M->getOrInsertFunction("mdmp_wait_many",
			   Type::getVoidTy(Ctx),
			   PointerType::getUnqual(Ctx),
			   I32Ty);

  auto Windows = analyseRequestWindows(PendingRequests, RegionEnd, AA, LI, M);

  MapVector<Instruction *, SmallVector<AsyncRequest *, 8>> GroupedWaits;

  for (RequestWindowInfo &Info : Windows) {
    SmallPtrSet<Instruction *, 4> UniqueWaitPoints;

    for (Instruction *InsertPt : Info.WaitPoints) {
      if (!DT.dominates(Info.Req->StartPoint, InsertPt)) {
        InsertPt = Info.Req->StartPoint->getParent()->getTerminator();
      }

      Instruction *HoistPt = InsertPt;
      Loop *L = LI.getLoopFor(HoistPt->getParent());
      Loop *ReqLoop = LI.getLoopFor(Info.Req->StartPoint->getParent());

      while (L && L != ReqLoop) {
        BasicBlock *Latch = L->getLoopLatch();
        if (Latch && DT.dominates(HoistPt->getParent(), Latch)) {
          BasicBlock *Preheader = L->getLoopPreheader();
          if (Preheader) {
            Instruction *PotentialHoistPt = Preheader->getTerminator();
            if (DT.dominates(Info.Req->StartPoint, PotentialHoistPt)) {
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

    for (Instruction *Pt : UniqueWaitPoints) {
      GroupedWaits[Pt].push_back(Info.Req);
    }
  }

  unsigned MaxBatchSize = 0;
  for (auto &KV : GroupedWaits) {
    MaxBatchSize = std::max<unsigned>(MaxBatchSize, KV.second.size());
  }

  AllocaInst *WaitIDsScratch = nullptr;
  ArrayType *WaitIDsScratchTy = nullptr;

  if (MaxBatchSize > 1) {
    Function *F = PendingRequests.front().StartPoint->getFunction();
    BasicBlock &EntryBB = F->getEntryBlock();
    IRBuilder<> EntryBuilder(&*EntryBB.getFirstInsertionPt());

    WaitIDsScratchTy = ArrayType::get(I32Ty, MaxBatchSize);
    WaitIDsScratch =
      EntryBuilder.CreateAlloca(WaitIDsScratchTy, nullptr, "mdmp_wait_ids_scratch");
  }

  for (auto &KV : GroupedWaits) {
    Instruction *InsertPt = KV.first;

    // Optional dedup by token source.
    SmallVector<AsyncRequest *, 8> UniqueReqs;
    SmallPtrSet<Value *, 8> SeenTokenSources;

    for (AsyncRequest *Req : KV.second) {
      Value *Key = Req->WaitTokenValue ? Req->WaitTokenValue
	: static_cast<Value *>(Req->WaitTokenAlloc);
      if (SeenTokenSources.insert(Key).second)
        UniqueReqs.push_back(Req);
    }

    IRBuilder<> Builder(InsertPt);

    if (UniqueReqs.size() == 1) {
      Value *WaitID =
	materialiseWaitTokenForUse(*UniqueReqs[0], InsertPt, I32Ty, Builder, DT);
      Builder.CreateCall(runtime_wait, {WaitID});
      continue;
    }

    Value *Zero = ConstantInt::get(I32Ty, 0);

    for (unsigned i = 0; i < UniqueReqs.size(); ++i) {
      Value *Idx = ConstantInt::get(I32Ty, i);
      Value *Slot =
	Builder.CreateInBoundsGEP(WaitIDsScratchTy, WaitIDsScratch, {Zero, Idx});

      Value *WaitID =
	materialiseWaitTokenForUse(*UniqueReqs[i], InsertPt, I32Ty, Builder, DT);

      Builder.CreateStore(WaitID, Slot);
    }

    Value *BasePtr =
      Builder.CreateInBoundsGEP(WaitIDsScratchTy, WaitIDsScratch, {Zero, Zero});

    Builder.CreateCall(runtime_wait_many,
                       {BasePtr, ConstantInt::get(I32Ty, UniqueReqs.size())});
  }

  PendingRequests.clear();
}


void MDMPPass::injectThrottledProgress(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI, Module *M) {

  if (PendingRequests.empty())
    return;

  LLVMContext &Ctx = M->getContext();
  IntegerType *I32Ty = Type::getInt32Ty(Ctx);

  FunctionCallee runtime_maybe_progress =
    M->getOrInsertFunction("mdmp_maybe_progress", Type::getVoidTy(Ctx));

  auto Windows = analyseRequestWindows(PendingRequests, nullptr, AA, LI, M);
  if (Windows.empty())
    return;

  SmallVector<Loop *, 16> LeafLoops;
  collectLeafLoops(LI, LeafLoops);
  if (LeafLoops.empty())
    return;

  static constexpr uint32_t ThrottleEvery = 64;
  static_assert((ThrottleEvery & (ThrottleEvery - 1)) == 0,
                "ThrottleEvery must be a power of two");

  ConstantInt *Zero = ConstantInt::get(I32Ty, 0);
  ConstantInt *One  = ConstantInt::get(I32Ty, 1);
  ConstantInt *Mask = ConstantInt::get(I32Ty, ThrottleEvery - 1);

  BasicBlock &EntryBB = F.getEntryBlock();
  Instruction *EntryIP = &*EntryBB.getFirstInsertionPt();
  IRBuilder<> EntryBuilder(EntryIP);

  AllocaInst *CounterAlloc =
    EntryBuilder.CreateAlloca(I32Ty, nullptr, "mdmp_progress_counter");
  EntryBuilder.CreateStore(Zero, CounterAlloc);

  SmallPtrSet<BasicBlock *, 16> InstrumentedHeaders;

  for (Loop *L : LeafLoops) {
    BasicBlock *Header = L->getHeader();

    if (!InstrumentedHeaders.insert(Header).second)
      continue;

    bool ShouldInstrument = false;
    for (const RequestWindowInfo &Info : Windows) {
      if (requestWindowCoversLoopHeader(Info, L, DT)) {
        ShouldInstrument = true;
        break;
      }
    }

    if (!ShouldInstrument)
      continue;

    Instruction *InsertPt = &*Header->getFirstInsertionPt();
    IRBuilder<> LoopBuilder(InsertPt);

    Value *Cur = LoopBuilder.CreateLoad(I32Ty, CounterAlloc, "mdmp_prog_cur");
    Value *Next = LoopBuilder.CreateAdd(Cur, One, "mdmp_prog_next");
    LoopBuilder.CreateStore(Next, CounterAlloc);

    Value *Masked = LoopBuilder.CreateAnd(Next, Mask, "mdmp_prog_masked");
    Value *DoTick = LoopBuilder.CreateICmpEQ(Masked, Zero, "mdmp_prog_tick");

    Instruction *ThenTerm = SplitBlockAndInsertIfThen(DoTick, InsertPt, false);
    IRBuilder<> ThenBuilder(ThenTerm);
    ThenBuilder.CreateCall(runtime_maybe_progress);
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
