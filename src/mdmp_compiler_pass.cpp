#include "mdmp_compiler_pass.h"

using namespace llvm;

bool mdmpEnvFlagEnabled(const char *Name, bool DefaultValue = false) {
  const char *V = std::getenv(Name);
  if (!V) return DefaultValue;

  if (std::strcmp(V, "0") == 0) return false;
  if (std::strcmp(V, "false") == 0) return false;
  if (std::strcmp(V, "FALSE") == 0) return false;
  if (std::strcmp(V, "off") == 0) return false;
  if (std::strcmp(V, "OFF") == 0) return false;

  return true;
}

unsigned mdmpEnvUnsigned(const char *Name, unsigned DefaultValue) {
  const char *V = std::getenv(Name);
  if (!V || *V == '\0') return DefaultValue;

  char *End = nullptr;
  unsigned long Parsed = std::strtoul(V, &End, 10);
  if (End == V || *End != '\0')
    return DefaultValue;

  // This is required to avoid division by zero bugs
  if (Parsed == 0)
    return 1;

  if (Parsed > std::numeric_limits<unsigned>::max())
    return DefaultValue;

  return static_cast<unsigned>(Parsed);
}

void MDMPPass::collectNonLeafLoops(Loop *L, SmallVectorImpl<Loop *> &Out) {
  if (!L->getSubLoops().empty())
    Out.push_back(L);

  for (Loop *SubL : L->getSubLoops()) {
    collectNonLeafLoops(SubL, Out);
  }
}

void MDMPPass::collectNonLeafLoops(LoopInfo &LI, SmallVectorImpl<Loop *> &Out) {
  for (Loop *TopL : LI) {
    collectNonLeafLoops(TopL, Out);
  }
}

Instruction *MDMPPass::mdmpInstructionAfter(Instruction *I) {
  auto It = I->getIterator();
  ++It;
  assert(It != I->getParent()->end() &&
         "Expected a non-terminator instruction");
  return &*It;
}

bool MDMPPass::requestWindowSuggestsCallSiteProgressRelaxed(const RequestWindowInfo &Info, Instruction *Inst, DominatorTree &DT) {                                                            
                                                            
  if (!DT.dominates(Info.Req->StartPoint, Inst))
    return false;

  // If this instruction is already in a known live block, it is an obvious
  // candidate for a progress poke.
  if (Info.LiveBlocks.contains(Inst->getParent()))
    return true;

  // If we have no known wait points, any substantial call after the request
  // start is a reasonable fallback.
  if (Info.WaitPoints.empty())
    return true;

  for (Instruction *WP : Info.WaitPoints) {
    // Same-block ordering.
    if (WP->getParent() == Inst->getParent()) {
      if (Inst->comesBefore(WP))
        return true;
    }

    // If this call dominates a later wait, then it lies on a path where the
    // request is still plausibly in flight.
    if (DT.dominates(Inst, WP))
      return true;
  }

  return false;
}

bool MDMPPass::isCandidateCallForProgress(Instruction *Inst) {
  auto *CB = dyn_cast<CallBase>(Inst);
  if (!CB)
    return false;

  if (CB->isInlineAsm())
    return false;

  if (isIgnorableIntrinsicForMDMP(Inst))
    return false;

  if (Function *CalledFn = CB->getCalledFunction()) {
    StringRef Name = CalledFn->getName();

    if (CalledFn->isIntrinsic())
      return false;

    if (isAsyncMDMPOpName(Name))
      return false;

    if (isHardBarrierCallName(Name))
      return false;

    if (Name == "mdmp_progress" || Name == "mdmp_maybe_progress")
      return false;

    if (Name == "mdmp_get_rank" || Name == "mdmp_get_size" || Name == "mdmp_wtime")
      return false;

    if (Name == "MPI_Comm_rank" || Name == "MPI_Comm_size")
      return false;

    if (isClearlyUnhelpfulProgressCallName(Name))
      return false;
  }

  // Keep indirect or unknown calls as possible candidates; they may represent
  // substantial work and we cannot cheaply classify them here.
  return true;
}

bool MDMPPass::isClearlyUnhelpfulProgressCallName(StringRef Name) {
  // Allocation / deallocation
  if (Name == "malloc" || Name == "calloc" || Name == "realloc" ||
      Name == "free" || Name == "aligned_alloc" || Name == "posix_memalign")
    return true;

  // Printing / logging / formatting
  if (Name == "printf" || Name == "fprintf" || Name == "sprintf" ||
      Name == "snprintf" || Name == "puts" || Name == "fputs" ||
      Name == "fwrite" || Name == "fflush" || Name == "perror")
    return true;

  // Timing / tiny utility helpers commonly seen in Gadget
  if (Name == "second" || Name == "timediff" || Name == "time" ||
      Name == "clock" || Name == "gettimeofday")
    return true;

  // Generic library helpers that are usually not where we want to spend
  // a scarce progress fallback site.
  if (Name == "qsort" || Name == "bsearch")
    return true;

  return false;
}

bool MDMPPass::isHardBarrierInstForWaitPlacement(Instruction *Inst) {
  if (auto *CB = dyn_cast<CallBase>(Inst)) {
    if (Function *CalledFn = CB->getCalledFunction()) {
      return isHardBarrierCallName(CalledFn->getName());
    }
  }
  return false;
}

bool MDMPPass::isAsyncMDMPInstForWaitPlacement(Instruction *Inst) {
  if (auto *CB = dyn_cast<CallBase>(Inst)) {
    if (Function *CalledFn = CB->getCalledFunction()) {
      return isAsyncMDMPOpName(CalledFn->getName());
    }
  }
  return false;
}

bool MDMPPass::getDirectIdentifiedAccessInfo(const MemoryLocation &Loc, const DataLayout &DL, const Value *&Owner, int64_t &ByteOffset) {
                                          
  int64_t Off = 0;
  const Value *Base = GetPointerBaseWithConstantOffset(Loc.Ptr, Off, DL);
  if (!Base)
    return false;

  const Value *U = getUnderlyingObject(Base);
  if (!isIdentifiedObject(U))
    return false;

  // Exclude dynamic-memory-style roots.
  if (isa<LoadInst>(U) || isa<Argument>(U) || isa<PHINode>(U) || isa<CallBase>(U))
    return false;

  Owner = U;
  ByteOffset = Off;
  return true;
}

bool MDMPPass::getLoadedFieldPointeeInfo(const MemoryLocation &Loc, const DataLayout &DL, const Value *&Owner, int64_t &FieldOffset) {
                                      
  int64_t Off = 0;
  const Value *Base = GetPointerBaseWithConstantOffset(Loc.Ptr, Off, DL);
  if (!Base)
    return false;

  const Value *U = getUnderlyingObject(Base);
  auto *LI = dyn_cast<LoadInst>(U);
  if (!LI)
    return false;

  int64_t FieldOff = 0;
  const Value *FieldBase =
      GetPointerBaseWithConstantOffset(LI->getPointerOperand(), FieldOff, DL);
  if (!FieldBase)
    return false;

  const Value *FieldOwner = getUnderlyingObject(FieldBase);
  if (!isIdentifiedObject(FieldOwner))
    return false;

  Owner = FieldOwner;
  FieldOffset = FieldOff;
  return true;
}

bool MDMPPass::instructionIsTrueConsumerOrClobber(Instruction *I, const TrackedBuffer &Buf, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {

  if (isIgnorableIntrinsicForMDMP(I))
    return false;

  if (isMDMPInternalAllocaAccess(I))
    return false;

  // TEMPORARY HACK. DO NOT KEEP AS IS NOT GENERALISABLE
  if (auto *CB = dyn_cast<CallBase>(I)) {
    if (Function *F = CB->getCalledFunction()) {
      StringRef Name = F->getName();
      
      // Memory Allocations
      if (Name.contains("malloc") || Name.contains("calloc") || Name.contains("realloc") || 
          Name.contains("free") || Name.contains("Znwm") || Name.contains("Znam") || 
          Name.contains("ZdlPv") || Name.contains("ZdaPv") || 
          Name.contains("Allocate") || Name.contains("Release")) { 
        return false;
      }
      
      // Safe Getters
      if (Name == "mdmp_get_rank" || Name == "mdmp_get_size" || Name == "mdmp_wtime" ||
          Name == "MPI_Comm_rank" || Name == "MPI_Comm_size") {
        return false;
      }

      // Specific C-style I/O (fflush, system, fread, fwrite)
      if (Name.contains("printf") || Name.contains("puts") || Name.contains("fprintf") ||
          Name == "fflush" || Name == "system" || Name == "fopen" || Name == "fclose" ||
          Name == "fread" || Name == "fwrite" || Name.contains("ostream") || 
          Name.contains("exit") || Name.contains("abort") || Name.contains("Abort")) {
        return false;
      }

      // Applies to Math, C-Utilities (qsort), and other functions
      if (Name.contains("sqrt") || Name.contains("cbrt") || Name.contains("fabs") || 
          Name.contains("sin") || Name.contains("cos") || Name.contains("pow") ||
          Name.contains("max") || Name.contains("min") || 
          Name == "qsort" || Name == "bsearch" || 
          Name.contains("evaluate") || Name.contains("factor") || Name.contains("distribute")) {
        
        bool pointerOverlaps = false;
        
        // Check every argument. If the function takes pointers 
        // this loop skips, and we deem it safe
        for (Value *Arg : CB->args()) {
          if (Arg->getType()->isPointerTy()) {
            MemoryLocation ArgLoc(Arg, LocationSize::beforeOrAfterPointer());
            
            if (areDefinitelyDisjoint(ArgLoc, Buf.Loc, DL)) continue;
            
            if (AA.alias(ArgLoc, Buf.Loc) != AliasResult::NoAlias) {
              pointerOverlaps = true;
              break;
            }
          }
        }
        
        if (!pointerOverlaps) {
          return false; // Safely bypass the Black Box!
        }
      }
    }

    // Indirect Call Bypass
    if (CB->isIndirectCall()) {
      if (CB->getType()->isPointerTy() && CB->arg_size() == 2) {
        if (CB->getArgOperand(0)->getType()->isPointerTy() && 
            CB->getArgOperand(1)->getType()->isIntegerTy()) { 
          return false; 
        }
      }
    }
  }
    
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    // For send-like buffers, local CPU reads are okay.
    if (Buf.isNetworkReadOnly)
      return false;

    return locationsMayOverlap(MemoryLocation::get(LI), Buf.Loc, AA, DL);
  }

  if (auto *SI = dyn_cast<StoreInst>(I)) {
    // Stores always clash if overlapping.
    return locationsMayOverlap(MemoryLocation::get(SI), Buf.Loc, AA, DL);
  }

  ModRefInfo MR = AA.getModRefInfo(I, Buf.Loc);
  if (!isModOrRefSet(MR))
    return false;

  MemoryAccess *MA = MSSA.getMemoryAccess(I);

  // Send-like network access:
  // only local writes/clobbers matter.
  if (Buf.isNetworkReadOnly) {
    if (!isModSet(MR))
      return false;

    // A pure MemoryUse is read-only wrt memory state, so safe for send-like buffers.
    if (MA && isa<MemoryUse>(MA))
      return false;

    return true;
  }

  // Recv-like network access:
  // both local reads and writes matter.
  if (isRefSet(MR))
    return true;
  if (isModSet(MR))
    return true;

  return false;
}

bool MDMPPass::functionContainsAsyncMDMPCall(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (Function *Callee = CB->getCalledFunction()) {
          if (isAsyncMDMPOpName(Callee->getName()))
            return true;
        }
      }
    }
  }
  return false;
}

bool MDMPPass::instructionTouchesAnyTrackedBufferPhase2(Instruction *I, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {
                                                     
  for (const TrackedBuffer &Buf : Buffers) {
    if (instructionIsTrueConsumerOrClobber(I, Buf, AA, MSSA, DL))
      return true;
  }
  return false;
}

Instruction *MDMPPass::findFirstTrueConflictInBlock(BasicBlock *BB, BasicBlock::iterator StartIt, Instruction *RegionEnd, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {
  auto sameBasePreciseOverlap = [&](const MemoryLocation &A,
                                    const MemoryLocation &B) -> bool {
    int64_t OffA = 0, OffB = 0;
    const Value *BaseA = GetPointerBaseWithConstantOffset(A.Ptr, OffA, DL);
    const Value *BaseB = GetPointerBaseWithConstantOffset(B.Ptr, OffB, DL);

    if (!BaseA || !BaseB)
      return false;

    const Value *UA = getUnderlyingObject(BaseA);
    const Value *UB = getUnderlyingObject(BaseB);
    if (UA != UB)
      return false;

    auto SA = getPreciseSizeBytes(A.Size);
    auto SB = getPreciseSizeBytes(B.Size);
    if (!SA || !SB)
      return false;

    int64_t AStart = OffA;
    int64_t AEnd   = OffA + static_cast<int64_t>(*SA);
    int64_t BStart = OffB;
    int64_t BEnd   = OffB + static_cast<int64_t>(*SB);

    return !(AEnd <= BStart || BEnd <= AStart);
  };

  auto asyncBufferSetsDefinitelyConflict =
    [&](ArrayRef<TrackedBuffer> A, ArrayRef<TrackedBuffer> B) -> bool {
      for (const TrackedBuffer &TA : A) {
	for (const TrackedBuffer &TB : B) {
	  if (areDefinitelyDisjoint(TA.Loc, TB.Loc, DL))
	    continue;

	  // Strong proof #1: same underlying object + precise overlapping ranges.
	  if (sameBasePreciseOverlap(TA.Loc, TB.Loc))
	    return true;

	  // Strong proof #2: LLVM proves exact aliasing.
	  if (AA.alias(TA.Loc, TB.Loc) == AliasResult::MustAlias)
	    return true;
	}
      }
      return false;
    };

  for (auto It = StartIt; It != BB->end(); ++It) {
    Instruction *Inst = &*It;

    if (RegionEnd && Inst == RegionEnd)
      return Inst;

    if (!Inst->mayReadOrWriteMemory())
      continue;

    if (isHardBarrierInstForWaitPlacement(Inst))
      return Inst;

    if (auto *CB = dyn_cast<CallBase>(Inst)) {
      if (Function *F = CB->getCalledFunction()) {
        if (F->getName().starts_with("__mdmp_marker_") ||
            F->getName() == "mdmp_get_rank" ||
            F->getName() == "mdmp_get_size" ||
            F->getName() == "mdmp_wtime") {
          // These are MDMP placeholders. We guarantee they do not
          // implicitly clobber unrelated memory buffers.
          continue;
        }
      }
    }

    // For later async MDMP ops, allow pipelining unless we have strong proof
    // that they really touch the same bytes.
    if (isAsyncMDMPInstForWaitPlacement(Inst)) {
      auto *CB = dyn_cast<CallBase>(Inst);

      if (!CB)
        return Inst;

      auto OtherBuffers = getTrackedBuffersForAsyncRuntimeCall(CB);

      // If we cannot reconstruct the async op's buffer set (e.g. mdmp_commit),
      // stay conservative.
      if (!OtherBuffers)
        return Inst;

      if (asyncBufferSetsDefinitelyConflict(*OtherBuffers, Buffers))
        return Inst;

      continue;
    }

    if (instructionTouchesAnyTrackedBufferPhase2(Inst, Buffers, AA, MSSA, DL)) {
      errs() << "[MDMP DEBUG] Conflict detected!\n";
      errs() << "    Instruction: " << *Inst << "\n";
      if (Inst->getParent()) {
        errs() << "    In Block:    " << Inst->getParent()->getName() << "\n";
        if (Inst->getFunction()) {
          errs() << "    In Function: " << Inst->getFunction()->getName() << "\n";
        }
      }
      if (auto *LI = dyn_cast<LoadInst>(Inst)) {
	errs() << "    Load Ptr:    " << *LI->getPointerOperand() << "\n";
	errs() << "    Underlying:  "
	       << *getUnderlyingObject(LI->getPointerOperand()) << "\n";
	if (const DebugLoc &Dbg = LI->getDebugLoc()) {
	  errs() << "    Debug Loc:   " << Dbg.getLine() << ":" << Dbg.getCol() << "\n";
	}
      }

      if (auto *SI = dyn_cast<StoreInst>(Inst)) {
	errs() << "    Store Ptr:   " << *SI->getPointerOperand() << "\n";
	errs() << "    Underlying:  "
	       << *getUnderlyingObject(SI->getPointerOperand()) << "\n";
	if (const DebugLoc &Dbg = SI->getDebugLoc()) {
	  errs() << "    Debug Loc:   " << Dbg.getLine() << ":" << Dbg.getCol() << "\n";
	}
      }

      for (const TrackedBuffer &Buf : Buffers) {
	errs() << "    Tracked Ptr: " << *Buf.Loc.Ptr << "\n";
	errs() << "    Buf Underlying: "
	       << *getUnderlyingObject(Buf.Loc.Ptr) << "\n";
      }

      return Inst;
    }
  }

  return nullptr;
}

bool MDMPPass::isIgnorableIntrinsicForMDMP(Instruction *I) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  if (!II) return false;

  switch (II->getIntrinsicID()) {
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::dbg_assign:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_label:
  case Intrinsic::dbg_value:
    return true;
  default:
    return false;
  }
}

bool MDMPPass::instructionConflictsWithTrackedBufferMSSA(Instruction *I, const TrackedBuffer &Buf, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {
                                                      
  if (isIgnorableIntrinsicForMDMP(I))
    return false;

  if (isMDMPInternalAllocaAccess(I))
    return false;

  if (auto *LI = dyn_cast<LoadInst>(I)) {
    // Local reads are okay while a send/read-only network op is in flight.
    if (Buf.isNetworkReadOnly)
      return false;

    return locationsMayOverlap(MemoryLocation::get(LI), Buf.Loc, AA, DL);
  }

  if (auto *SI = dyn_cast<StoreInst>(I)) {
    return locationsMayOverlap(MemoryLocation::get(SI), Buf.Loc, AA, DL);
  }

  // For non-load/store memory ops, use MemorySSA to distinguish
  // read-only accesses from true defs/writes.
  ModRefInfo MR = AA.getModRefInfo(I, Buf.Loc);

  if (Buf.isNetworkReadOnly) {
    // Send-like buffer: only local writes/clobbers matter.
    if (!isModSet(MR))
      return false;

    if (MemoryAccess *MA = MSSA.getMemoryAccess(I)) {
      if (isa<MemoryUse>(MA))
        return false; // read-only op, safe for in-flight send
    }

    return true;
  }

  // Recv-like buffer: local reads and writes both matter.
  if (!isModOrRefSet(MR))
    return false;

  return true;
}

bool MDMPPass::instructionConflictsWithAnyTrackedBufferMSSA(Instruction *I, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {

  if (isMDMPInternalAllocaAccess(I))
    return false;
  
  for (const TrackedBuffer &Buf : Buffers) {
    if (instructionConflictsWithTrackedBufferMSSA(I, Buf, AA, MSSA, DL))
      return true;
  }
  return false;
}

bool MDMPPass::waitTokenValueDominates(Value *V, Instruction *InsertPt, DominatorTree &DT) {
                                    
  if (!V) return false;

  if (isa<Constant>(V) || isa<Argument>(V) || isa<GlobalValue>(V))
    return true;

  if (auto *I = dyn_cast<Instruction>(V))
    return DT.dominates(I, InsertPt);

  return true;
}

Value *MDMPPass::materialiseWaitTokenForUse(const AsyncRequest &Req, Instruction *InsertPt, IntegerType *I32Ty, IRBuilder<> &Builder, DominatorTree &DT) {

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
     Name == "__mdmp_marker_commregion_end");

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

bool MDMPPass::isMDMPInternalAllocaAccess(Instruction *I) {
  Value *Ptr = nullptr;

  if (auto *LI = dyn_cast<LoadInst>(I))
    Ptr = LI->getPointerOperand();
  else if (auto *SI = dyn_cast<StoreInst>(I))
    Ptr = SI->getPointerOperand();
  else
    return false;

  const Value *Obj = getUnderlyingObject(Ptr);
  auto *AI = dyn_cast<AllocaInst>(Obj);
  if (!AI)
    return false;

  StringRef N = AI->getName();
  return N == "mdmp_req_token" ||
    N == "mdmp_wait_ids_scratch" ||
    N == "mdmp_progress_counter";
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

bool MDMPPass::requestWindowSuggestsLoopProgressRelaxed(const RequestWindowInfo &Info,
                                                        Loop *L,
                                                        DominatorTree &DT) {
  BasicBlock *Header = L->getHeader();
  Instruction *HeaderIP = &*Header->getFirstInsertionPt();

  // The request must definitely be live before the loop header.
  if (!DT.dominates(Info.Req->StartPoint, HeaderIP))
    return false;

  // If exact matching already says the loop header is in the live region,
  // then this is obviously a valid place.
  if (Info.LiveBlocks.contains(Header))
    return true;

  // If the request has no known wait points yet, any loop after the start point
  // is a reasonable fallback progress site.
  if (Info.WaitPoints.empty())
    return true;

  // Relaxed heuristic:
  // if the loop dominates a future wait point, or the wait point is inside the
  // loop body, then this loop lies on a path where the request remains in flight.
  for (Instruction *WP : Info.WaitPoints) {
    if (L->contains(WP->getParent()))
      return true;

    Instruction *HeaderIP = &*Header->getFirstInsertionPt();
    if (DT.dominates(HeaderIP, WP))
      return true;
  }

  return false;
}

bool MDMPPass::loopMayConflictWithTrackedBuffers(Loop *L, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {                                                                                                                                       
  if (!L)
    return true;

  auto sameBasePreciseOverlap = [&](const MemoryLocation &A,
                                    const MemoryLocation &B) -> bool {
    int64_t OffA = 0, OffB = 0;
    const Value *BaseA = GetPointerBaseWithConstantOffset(A.Ptr, OffA, DL);
    const Value *BaseB = GetPointerBaseWithConstantOffset(B.Ptr, OffB, DL);

    if (!BaseA || !BaseB)
      return false;

    const Value *UA = getUnderlyingObject(BaseA);
    const Value *UB = getUnderlyingObject(BaseB);
    if (UA != UB)
      return false;

    auto SA = getPreciseSizeBytes(A.Size);
    auto SB = getPreciseSizeBytes(B.Size);
    if (!SA || !SB)
      return false;

    int64_t AStart = OffA;
    int64_t AEnd   = OffA + static_cast<int64_t>(*SA);
    int64_t BStart = OffB;
    int64_t BEnd   = OffB + static_cast<int64_t>(*SB);

    return !(AEnd <= BStart || BEnd <= AStart);
  };

  auto asyncBufferSetsDefinitelyConflict =
    [&](ArrayRef<TrackedBuffer> A, ArrayRef<TrackedBuffer> B) -> bool {
      for (const TrackedBuffer &TA : A) {
	for (const TrackedBuffer &TB : B) {
	  if (areDefinitelyDisjoint(TA.Loc, TB.Loc, DL))
	    continue;

	  // Strong proof #1: same underlying object + precise overlapping ranges.
	  if (sameBasePreciseOverlap(TA.Loc, TB.Loc))
	    return true;

	  // Strong proof #2: LLVM proves exact aliasing.
	  if (AA.alias(TA.Loc, TB.Loc) == AliasResult::MustAlias)
	    return true;
	}
      }
      return false;
    };

  for (BasicBlock *LoopBB : L->getBlocksVector()) {
    for (Instruction &I : *LoopBB) {
      Instruction *Inst = &I;

      if (!Inst->mayReadOrWriteMemory())
        continue;

      if (isIgnorableIntrinsicForMDMP(Inst))
        continue;

      // Any hard barrier inside the loop means we should not carry the request
      // around the backedge.
      if (isHardBarrierInstForWaitPlacement(Inst))
        return true;

      // For later async MDMP ops, allow pipelining unless we have strong proof
      // that they really touch the same bytes.
      if (isAsyncMDMPInstForWaitPlacement(Inst)) {
        auto *CB = dyn_cast<CallBase>(Inst);
        if (!CB)
          return true;

        auto OtherBuffers = getTrackedBuffersForAsyncRuntimeCall(CB);

        // If we cannot reconstruct the async op's buffer set (e.g. mdmp_commit),
        // stay conservative.
        if (!OtherBuffers)
          return true;

        if (asyncBufferSetsDefinitelyConflict(*OtherBuffers, Buffers))
          return true;

        continue;
      }

      if (instructionTouchesAnyTrackedBufferPhase2(Inst, Buffers, AA, MSSA, DL))
        return true;
    }
  }

  return false;
}

bool MDMPPass::shouldForceWaitAtLoopBackedge(Loop *EdgeLoop, Loop *ReqLoop, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL) {
                                             
  if (!EdgeLoop)
    return false;

  // Never carry requests across iterations of the loop that issued them,
  // or any enclosing parent loop around that issuing loop.
  for (Loop *L = ReqLoop; L; L = L->getParentLoop()) {
    if (L == EdgeLoop)
      return true;
  }

  // Otherwise, only force a wait if the loop body can actually consume/clobber
  // one of the tracked buffers.
  return loopMayConflictWithTrackedBuffers(EdgeLoop, Buffers, AA, MSSA, DL);
}


SmallVector<MDMPPass::RequestWindowInfo, 8> MDMPPass::analyseRequestWindows(ArrayRef<AsyncRequest> Requests, Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, MemorySSA &MSSA, Module *M) {                      

  const DataLayout &DL = M->getDataLayout();
  SmallVector<RequestWindowInfo, 8> Infos;
  Infos.reserve(Requests.size());

  for (const AsyncRequest &Req : Requests) {
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
      if (Instruction *Conflict =
	  findFirstTrueConflictInBlock(BB, State.StartIt, RegionEnd,
				       Req.Buffers, AA, MSSA, DL)) {
        Info.WaitPoints.push_back(Conflict);
        FoundStop = true;
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
          if (shouldForceWaitAtLoopBackedge(EdgeLoop, ReqLoop,
                                            Req.Buffers, AA, MSSA, DL)) {
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

MDMPPass::TrackedBuffer MDMPPass::makePreciseTrackedBuffer(Value *Ptr, Value *CountV, Value *TypeCodeV, Value *BytesV, bool IsNetworkReadOnly) {
  return { MemoryLocation(Ptr, derivePreciseSpan(CountV, TypeCodeV, BytesV)),
    IsNetworkReadOnly };
}

MDMPPass::TrackedBuffer MDMPPass::makeUnknownTrackedBuffer(Value *Ptr, bool IsNetworkReadOnly) {
  return { MemoryLocation(Ptr, LocationSize::beforeOrAfterPointer()),
    IsNetworkReadOnly };
}

std::optional<uint64_t> MDMPPass::getPreciseSizeBytes(LocationSize S) {
  if (!S.hasValue() || !S.isPrecise())
    return std::nullopt;
  return S.getValue();
}

bool MDMPPass::areDefinitelyDisjoint(const MemoryLocation &A, const MemoryLocation &B, const DataLayout &DL) {
  int64_t OffA = 0, OffB = 0;
  const Value *BaseA = GetPointerBaseWithConstantOffset(A.Ptr, OffA, DL);
  const Value *BaseB = GetPointerBaseWithConstantOffset(B.Ptr, OffB, DL);

  if (!BaseA || !BaseB)
    return false;

  const Value *UA = getUnderlyingObject(BaseA);
  const Value *UB = getUnderlyingObject(BaseB);

  const Value *DirectOwnerA = nullptr, *DirectOwnerB = nullptr;
  const Value *FieldOwnerA = nullptr, *FieldOwnerB = nullptr;
  int64_t DirectOffA = 0, DirectOffB = 0;
  int64_t FieldOffA = 0, FieldOffB = 0;
  
  bool ADirect = getDirectIdentifiedAccessInfo(A, DL, DirectOwnerA, DirectOffA);
  bool BDirect = getDirectIdentifiedAccessInfo(B, DL, DirectOwnerB, DirectOffB);
  
  bool AField = getLoadedFieldPointeeInfo(A, DL, FieldOwnerA, FieldOffA);
  bool BField = getLoadedFieldPointeeInfo(B, DL, FieldOwnerB, FieldOffB);
  
  // Rule A: direct aggregate metadata vs pointee loaded from its field
  if (ADirect && BField && DirectOwnerA == FieldOwnerB)
    return true;
  if (BDirect && AField && DirectOwnerB == FieldOwnerA)
    return true;
  
  // Rule B: different loaded fields of same aggregate
  if (AField && BField &&
      FieldOwnerA == FieldOwnerB &&
      FieldOffA != FieldOffB)
    return true;
  
  // Rule C: separate local identified object vs pointee loaded from different aggregate
  if (ADirect && BField &&
    DirectOwnerA != FieldOwnerB &&
    isa<AllocaInst>(DirectOwnerA))
    return true;
  if (BDirect && AField &&
      DirectOwnerB != FieldOwnerA &&
      isa<AllocaInst>(DirectOwnerB))
    return true;
    
  // If the IR contains two different load instructions that load from the 
  // exact same struct field (e.g., domain.commDataRecv), they represent the 
  // exact same underlying heap array. We must unify them here!
  if (UA != UB) {
    if (auto *LA = dyn_cast<LoadInst>(UA)) {
      if (auto *LB = dyn_cast<LoadInst>(UB)) {
        int64_t OffsetA = 0, OffsetB = 0;
        const Value *BasePtrA = GetPointerBaseWithConstantOffset(LA->getPointerOperand(), OffsetA, DL);
        const Value *BasePtrB = GetPointerBaseWithConstantOffset(LB->getPointerOperand(), OffsetB, DL);
        
        if (BasePtrA && BasePtrB && BasePtrA == BasePtrB && OffsetA == OffsetB) {
          UB = UA; // Treat them as the exact same object
        }
      }
    }
  }

  if (UA != UB && isIdentifiedObject(UA) && isIdentifiedObject(UB))
    return true;

  if (UA == UB) {
    auto SA = getPreciseSizeBytes(A.Size);
    auto SB = getPreciseSizeBytes(B.Size);
    if (!SA || !SB)
      return false; // Falls through to Alias Analysis

    int64_t AStart = OffA;
    int64_t AEnd   = OffA + static_cast<int64_t>(*SA);
    int64_t BStart = OffB;
    int64_t BEnd   = OffB + static_cast<int64_t>(*SB);

    return (AEnd <= BStart) || (BEnd <= AStart);
  }

  return false;
}

bool MDMPPass::locationsMayOverlap(const MemoryLocation &A, const MemoryLocation &B, AAResults &AA, const DataLayout &DL) {
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

bool MDMPPass::operandsAvailableBefore(CallInst *CI, Instruction *InsertBefore, DominatorTree &DT) {

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

bool MDMPPass::instructionConflictsWithTrackedBuffer(Instruction *I, const TrackedBuffer &Buf, AAResults &AA, const DataLayout &DL) {

  if (isMDMPInternalAllocaAccess(I))
    return false;
  
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

bool MDMPPass::instructionConflictsWithAnyTrackedBuffer(Instruction *I, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, const DataLayout &DL) {

  if (isMDMPInternalAllocaAccess(I))
    return false;
  
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

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildSendRecvBuffers(Value *Buf, Value *Count, Value *Type, Value *Bytes, bool IsSend) {
  return { makePreciseTrackedBuffer(Buf, Count, Type, Bytes, IsSend) };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildReduceBuffers(Value *SendBuf, Value *RecvBuf, Value *Count, Value *Type, Value *Bytes) {
  return {
    makePreciseTrackedBuffer(RecvBuf, Count, Type, Bytes, false),
    makePreciseTrackedBuffer(SendBuf, Count, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildGatherBuffers(Value *SendBuf, Value *SendCount, Value *RecvBuf, Value *Type, Value *Bytes) {
  // Send side is exact. Receive side depends on root/global_size, so keep
  // conservative unless your frontend can pass total recv-buffer bytes.
  return {
    makeUnknownTrackedBuffer(RecvBuf, false),
    makePreciseTrackedBuffer(SendBuf, SendCount, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildAllreduceBuffers(Value *SendBuf, Value *RecvBuf, Value *Count, Value *Type, Value *Bytes) {
  return {
    makePreciseTrackedBuffer(RecvBuf, Count, Type, Bytes, false),
    makePreciseTrackedBuffer(SendBuf, Count, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildAllgatherBuffers(Value *SendBuf, Value *Count, Value *RecvBuf, Value *Type, Value *Bytes) {
  return {
    makeUnknownTrackedBuffer(RecvBuf, false),
    makePreciseTrackedBuffer(SendBuf, Count, Type, Bytes, true)
  };
}

std::vector<MDMPPass::TrackedBuffer> MDMPPass::buildBcastBuffers(Value *Buf, Value *Count, Value *Type, Value *Bytes) {
								 
  // Conservatively treat as recv-like because non-root ranks get writes.
  return {
    makePreciseTrackedBuffer(Buf, Count, Type, Bytes, false)
  };
}


std::optional<std::vector<MDMPPass::TrackedBuffer>> MDMPPass::getTrackedBuffersForAsyncRuntimeCall(CallBase *CB) {
  if (!CB)
    return std::nullopt;

  Function *F = CB->getCalledFunction();
  if (!F)
    return std::nullopt;

  StringRef Name = F->getName();

  if (Name == "mdmp_send" || Name == "mdmp_recv" ||
      Name == "mdmp_register_send" || Name == "mdmp_register_recv") {
    bool IsSend = (Name.contains("send") && !Name.contains("recv"));
    return buildSendRecvBuffers(CB->getArgOperand(0),  // buf
                                CB->getArgOperand(1),  // count
                                CB->getArgOperand(2),  // type
                                CB->getArgOperand(3),  // bytes
                                IsSend);
  }

  if (Name == "mdmp_reduce" || Name == "mdmp_register_reduce") {
    return buildReduceBuffers(CB->getArgOperand(0),  // sendbuf
                              CB->getArgOperand(1),  // recvbuf
                              CB->getArgOperand(2),  // count
                              CB->getArgOperand(3),  // type
                              CB->getArgOperand(4)); // bytes
  }

  if (Name == "mdmp_gather" || Name == "mdmp_register_gather") {
    return buildGatherBuffers(CB->getArgOperand(0),  // sendbuf
                              CB->getArgOperand(1),  // sendcount
                              CB->getArgOperand(2),  // recvbuf
                              CB->getArgOperand(3),  // type
                              CB->getArgOperand(4)); // bytes
  }

  if (Name == "mdmp_allreduce" || Name == "mdmp_register_allreduce") {
    return buildAllreduceBuffers(CB->getArgOperand(0),
                                 CB->getArgOperand(1),
                                 CB->getArgOperand(2),
                                 CB->getArgOperand(3),
                                 CB->getArgOperand(4));
  }

  if (Name == "mdmp_allgather" || Name == "mdmp_register_allgather") {
    return buildAllgatherBuffers(CB->getArgOperand(0),
                                 CB->getArgOperand(1),
                                 CB->getArgOperand(2),
                                 CB->getArgOperand(3),
                                 CB->getArgOperand(4));
  }

  if (Name == "mdmp_bcast" || Name == "mdmp_register_bcast") {
    return buildBcastBuffers(CB->getArgOperand(0),
                             CB->getArgOperand(1),
                             CB->getArgOperand(2),
                             CB->getArgOperand(3));
  }

  // mdmp_commit cannot be reconstructed from call operands alone.
  if (Name == "mdmp_commit")
    return std::nullopt;

  return std::nullopt;
}

bool MDMPPass::trackedBufferSetsMayOverlap(ArrayRef<TrackedBuffer> A, ArrayRef<TrackedBuffer> B, AAResults &AA, const DataLayout &DL) {                                                                                      
                                           
  for (const TrackedBuffer &TA : A) {
    for (const TrackedBuffer &TB : B) {
      if (locationsMayOverlap(TA.Loc, TB.Loc, AA, DL))
        return true;
    }
  }
  return false;
}

bool MDMPPass::inlineThinMDMPWrappers(Module &M) {
  bool Changed = false;
  InlineFunctionInfo IFI;
  bool LocalChanged;

  unsigned IterationCount = 0;
  const unsigned MaxIterations = 6;

  do {
    LocalChanged = false;
    std::vector<CallBase *> CallsToInline;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;

          Function *Callee = CB->getCalledFunction();
          if (!Callee || Callee->isDeclaration() || Callee == &F)
            continue;

	  if (functionContainsAsyncMDMPCall(*Callee)) {
	    errs() << "[MDMP INLINE] inlining wrapper call to " << Callee->getName()
		   << " into " << F.getName() << "\n";
	    CallsToInline.push_back(CB);
	  }
        }
      }
    }

    for (CallBase *CB : CallsToInline) {
      InlineResult IR = InlineFunction(*CB, IFI);
      if (IR.isSuccess()) {
        LocalChanged = true;
        Changed = true;
      }
    }

    ++IterationCount;
  } while (LocalChanged && IterationCount < MaxIterations);

  return Changed;
}

PreservedAnalyses MDMPPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool changed = false;
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  NextProgressSiteID = 0;
  
  // Programmatically erase interprocedural wrapper boundaries
  changed |= inlineThinMDMPWrappers(M);

  for (auto &F : M) {
    if (!F.isDeclaration()) {
      // Re-fetch analyses (in case our inlining invalidated previous IR state)
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
  CompletedRegions.clear();

  bool Changed = transformFunctionsToCalls(F, AA, DT, LI);

  if (PendingRequests.empty() && CompletedRegions.empty())
    return Changed;

  LLVMContext &Ctx = F.getContext();
  Module *M = F.getParent();

  // Keep a copy for throttled progress injection.
  std::vector<AsyncRequest> AllRequestsForProgress;
  for (auto &R : CompletedRegions) {
    AllRequestsForProgress.insert(AllRequestsForProgress.end(),
                                  R.Requests.begin(), R.Requests.end());
  }
  AllRequestsForProgress.insert(AllRequestsForProgress.end(),
                                PendingRequests.begin(), PendingRequests.end());

  // Build MemorySSA on the transformed function for wait insertion.
  MemorySSA WaitMSSA(F, &AA, &DT);

  for (auto &R : CompletedRegions) {
    injectWaitsForRegion(R.Requests, R.RegionEnd, AA, LI, Ctx, M, DT, WaitMSSA);
  }

  if (!PendingRequests.empty()) {
    injectWaitsForRegion(PendingRequests, nullptr, AA, LI, Ctx, M, DT, WaitMSSA);
  }

  // Rebuild MemorySSA after wait insertion. Wait calls are real memory-affecting
  // barriers, so progress placement must see the post-wait IR, not the stale one.
  MemorySSA ProgressMSSA(F, &AA, &DT);

  bool EnableJITProgress = mdmpEnvFlagEnabled("MDMP_ENABLE_JIT_PROGRESS", true);

  if (EnableJITProgress && !AllRequestsForProgress.empty()) {
    injectThrottledProgress(AllRequestsForProgress, F, AA, DT, LI, ProgressMSSA, M);
  }

  CompletedRegions.clear();
  PendingRequests.clear();
  return Changed;
}


bool MDMPPass::transformFunctionsToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI) {
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

  std::vector<Instruction*> toDelete;
  bool Changed = false;
  struct DeclarativePendingOp {
    std::vector<TrackedBuffer> Buffers;
  };
  
  std::vector<DeclarativePendingOp> ActiveDeclarativeLocs;

  auto registerAsyncRequest = [&](CallInst *NewCall, std::vector<TrackedBuffer> BuffersToTrack) {
    AsyncRequest Req;
    Req.WaitTokenValue = NewCall;
    Req.StartPoint = NewCall;
    Req.Buffers = std::move(BuffersToTrack);

    // Create an Alloca in the Entry Block initialized to -1 (MDMP_PROCESS_NOT_INVOLVED)
    BasicBlock &EntryBB = F.getEntryBlock();
    IRBuilder<> EntryBuilder(&*EntryBB.getFirstInsertionPt());
    AllocaInst *Alloc = EntryBuilder.CreateAlloca(Type::getInt32Ty(Ctx), nullptr, "mdmp_req_token");
    EntryBuilder.CreateStore(ConstantInt::getSigned(Type::getInt32Ty(Ctx), -1), Alloc);

    // Store the true token immediately after the async call
    Instruction *NextI = mdmpInstructionAfter(NewCall);
    IRBuilder<> StoreBuilder(NextI);
    StoreBuilder.CreateStore(NewCall, Alloc);

    Req.WaitTokenAlloc = Alloc;
    PendingRequests.push_back(std::move(Req));
  };
 
  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (BasicBlock *BB : RPOT) { 
    //for (auto &BB : F) {
    for (auto &I : *BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->getCalledFunction()) continue;
            
      StringRef Name = CI->getCalledFunction()->getName();
      IRBuilder<> Builder(CI);

      if (Name == "__mdmp_marker_commregion_begin") {
	Changed = true;
        CallInst *NewCall = Builder.CreateCall(runtime_begin);
        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.clear(); 
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_send" || Name == "__mdmp_marker_recv") {
	Changed = true;
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
	hoistInitiation(NewCall, TrackedLocs, AA, DT);

	registerAsyncRequest(NewCall, TrackedLocs);

        toDelete.push_back(CI);
      } 
      else if (Name == "__mdmp_marker_register_send" || Name == "__mdmp_marker_register_recv") {
	Changed = true;
        Value *BufferPtr = CI->getArgOperand(0);
        Value *CountVal  = CI->getArgOperand(1);
        Value *TypeVal   = CI->getArgOperand(2);
        Value *ByteSize  = CI->getArgOperand(3);
        Value *ActorRank = CI->getArgOperand(4);
        Value *PeerRank  = CI->getArgOperand(5);
        Value *TagVal    = CI->getArgOperand(6); 

        bool isSend = (Name == "__mdmp_marker_register_send");
        std::vector<TrackedBuffer> TrackedLocs = buildSendRecvBuffers(BufferPtr, CountVal, TypeVal, ByteSize, isSend);

	CallInst *NewCall = Builder.CreateCall(isSend ? runtime_register_send : runtime_register_recv,
                                               {BufferPtr, CountVal, TypeVal, ByteSize, ActorRank, PeerRank, TagVal});

        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.push_back({TrackedLocs});
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_register_reduce") {
	Changed = true;
        Value *InBuf = CI->getArgOperand(0); Value *OutBuf = CI->getArgOperand(1);
        Value *ByteSize = CI->getArgOperand(4); 

	std::vector<TrackedBuffer> TrackedLocs = buildReduceBuffers(InBuf, OutBuf,
								    CI->getArgOperand(2),  // count
								    CI->getArgOperand(3),  // type
								    CI->getArgOperand(4)); // bytes

	CallInst *NewCall = Builder.CreateCall(runtime_register_reduce,
                                               {InBuf, OutBuf, CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
	
        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.push_back({TrackedLocs});

        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_register_gather") {
	Changed = true;
        Value *SendBuf = CI->getArgOperand(0); Value *RecvBuf = CI->getArgOperand(2);
        Value *ByteSize = CI->getArgOperand(4); 

        std::vector<TrackedBuffer> TrackedLocs = buildGatherBuffers(SendBuf,
                                                                    CI->getArgOperand(1),  // sendcount
                                                                    RecvBuf,
                                                                    CI->getArgOperand(3),  // type
                                                                    CI->getArgOperand(4)); // bytes

        CallInst *NewCall = Builder.CreateCall(runtime_register_gather,
                                               {SendBuf, CI->getArgOperand(1), RecvBuf, CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});

        CI->replaceAllUsesWith(NewCall);
        ActiveDeclarativeLocs.push_back({TrackedLocs});
	toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_commit") {
	Changed = true;
        CallInst *NewCommit = Builder.CreateCall(runtime_commit);
        CI->replaceAllUsesWith(NewCommit);

        // Build the full mixed hazard set for this declarative batch.
        std::vector<TrackedBuffer> CommitBuffers;
        size_t TotalTracked = 0;
        for (auto &Op : ActiveDeclarativeLocs)
          TotalTracked += Op.Buffers.size();

        CommitBuffers.reserve(TotalTracked);
        for (auto &Op : ActiveDeclarativeLocs) {
          CommitBuffers.insert(CommitBuffers.end(),
                               Op.Buffers.begin(), Op.Buffers.end());
        }

        if (!CommitBuffers.empty()) {
          hoistInitiation(NewCommit, CommitBuffers, AA, DT);
        }
       
        if (!CommitBuffers.empty()) {
	  registerAsyncRequest(NewCommit, CommitBuffers);
        }

        ActiveDeclarativeLocs.clear();
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_reduce") {
	Changed = true;
        CallInst *NewCall = Builder.CreateCall(runtime_reduce, 
					       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5), CI->getArgOperand(6)});
       
        std::vector<TrackedBuffer> Locs =
	  buildReduceBuffers(CI->getArgOperand(0),  // sendbuf
			     CI->getArgOperand(1),  // recvbuf
			     CI->getArgOperand(2),  // count
			     CI->getArgOperand(3),  // type
			     CI->getArgOperand(4)); // bytes 

	hoistInitiation(NewCall, Locs, AA, DT);

	CI->replaceAllUsesWith(NewCall);
	registerAsyncRequest(NewCall, Locs);
	
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_gather") {
	Changed =  true;
        CallInst *NewCall = Builder.CreateCall(runtime_gather, 
					       {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});
        
        std::vector<TrackedBuffer> Locs =
	  buildGatherBuffers(CI->getArgOperand(0),  // sendbuf
			     CI->getArgOperand(1),  // sendcount
			     CI->getArgOperand(2),  // recvbuf
			     CI->getArgOperand(3),  // type
			     CI->getArgOperand(4)); // bytes
	hoistInitiation(NewCall, Locs, AA, DT);

	CI->replaceAllUsesWith(NewCall);
	registerAsyncRequest(NewCall, Locs);

        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_allreduce" || Name == "__mdmp_marker_register_allreduce") {
	Changed = true;
        FunctionCallee target_func = (Name == "__mdmp_marker_allreduce") ? runtime_allreduce : runtime_register_allreduce;
        CallInst *NewCall = Builder.CreateCall(target_func,
                                               {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4), CI->getArgOperand(5)});

        std::vector<TrackedBuffer> Locs =
          buildAllreduceBuffers(CI->getArgOperand(0),  // sendbuf
                                CI->getArgOperand(1),  // recvbuf
                                CI->getArgOperand(2),  // count
                                CI->getArgOperand(3),  // type
                                CI->getArgOperand(4)); // bytes

        CI->replaceAllUsesWith(NewCall);

        if (Name == "__mdmp_marker_allreduce") {
          hoistInitiation(NewCall, Locs, AA, DT);
	  registerAsyncRequest(NewCall, Locs);
        } else {
          ActiveDeclarativeLocs.push_back({Locs});
        }

        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_allgather" || Name == "__mdmp_marker_register_allgather") {
	Changed = true;
        FunctionCallee target_func = (Name == "__mdmp_marker_allgather") ? runtime_allgather : runtime_register_allgather;
        CallInst *NewCall = Builder.CreateCall(target_func,
                                               {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});

        std::vector<TrackedBuffer> Locs =
          buildAllgatherBuffers(CI->getArgOperand(0),  // sendbuf
                                CI->getArgOperand(1),  // count
                                CI->getArgOperand(2),  // recvbuf
                                CI->getArgOperand(3),  // type
                                CI->getArgOperand(4)); // bytes

        CI->replaceAllUsesWith(NewCall);

        if (Name == "__mdmp_marker_allgather") {
          hoistInitiation(NewCall, Locs, AA, DT);
	  registerAsyncRequest(NewCall, Locs);
        } else {
          ActiveDeclarativeLocs.push_back({Locs});
        }

        toDelete.push_back(CI);
      }

      else if (Name == "__mdmp_marker_bcast" || Name == "__mdmp_marker_register_bcast") {
	Changed = true;
        FunctionCallee target_func = (Name == "__mdmp_marker_bcast") ? runtime_bcast : runtime_register_bcast;
        CallInst *NewCall = Builder.CreateCall(target_func,
                                               {CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});

        std::vector<TrackedBuffer> Locs =
          buildBcastBuffers(CI->getArgOperand(0),  // buffer
                            CI->getArgOperand(1),  // count
                            CI->getArgOperand(2),  // type
                            CI->getArgOperand(3)); // bytes

        CI->replaceAllUsesWith(NewCall);

        if (Name == "__mdmp_marker_bcast") {
          hoistInitiation(NewCall, Locs, AA, DT);
	  registerAsyncRequest(NewCall, Locs);
        } else {
          ActiveDeclarativeLocs.push_back({Locs});
        }

        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_commregion_end") {
	Changed = true;
	CallInst *NewEnd = Builder.CreateCall(runtime_end);
	
	if (!PendingRequests.empty()) {
	  CompletedRegion R;
	  R.RegionEnd = NewEnd;
	  R.Requests = PendingRequests;
	  CompletedRegions.push_back(std::move(R));
	  PendingRequests.clear();
	}
	
	toDelete.push_back(CI);
      }      
      else if (Name == "__mdmp_marker_get_rank") { Changed = true; CallInst *NewCall = Builder.CreateCall(runtime_get_rank); CI->replaceAllUsesWith(NewCall); toDelete.push_back(CI); }
      else if (Name == "__mdmp_marker_get_size") { Changed = true; CallInst *NewCall = Builder.CreateCall(runtime_get_size); CI->replaceAllUsesWith(NewCall); toDelete.push_back(CI); }
      else if (Name == "__mdmp_marker_init") { Changed = true; Builder.CreateCall(runtime_init); toDelete.push_back(CI); }
      else if (Name == "__mdmp_marker_final") { Changed = true; Builder.CreateCall(runtime_final); toDelete.push_back(CI); }
      else if (Name == "__mdmp_marker_sync") { Changed = true; Builder.CreateCall(runtime_sync); toDelete.push_back(CI); }
      else if (Name == "__mdmp_marker_wtime") { Changed = true; CallInst *NewCall = Builder.CreateCall(runtime_wtime); CI->replaceAllUsesWith(NewCall); toDelete.push_back(CI); }
      else if (Name == "__mdmp_marker_set_debug") {
	Changed = true;
        Value *EnableArg = CI->getArgOperand(0); 
        CallInst *NewCall = Builder.CreateCall(runtime_set_debug, {EnableArg});
        CI->replaceAllUsesWith(NewCall);
        toDelete.push_back(CI);
      }
      else if (Name == "__mdmp_marker_abort") {
	Changed = true; 
        Value *EnableArg = CI->getArgOperand(0);
        CallInst *NewCall = Builder.CreateCall(runtime_abort, {EnableArg});
        CI->replaceAllUsesWith(NewCall);
        toDelete.push_back(CI);
      }
    }
  }
  for (Instruction *I : toDelete) I->eraseFromParent();
  return Changed;
}

void MDMPPass::hoistInitiation(CallInst *CI, std::vector<TrackedBuffer> &Buffers, AAResults &AA, DominatorTree &DT) {
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

    if (isHardMotionBarrier(PredTerm))
      break;

    InsertBefore = PredTerm;
    CurBB = Pred;
  }

  if (InsertBefore != CI)
    CI->moveBefore(InsertBefore->getIterator());
}

void MDMPPass::injectWaitsForRegion(ArrayRef<AsyncRequest> Requests, Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, LLVMContext &Ctx, Module *M, DominatorTree &DT, MemorySSA &MSSA) {
                                    
  IntegerType *I32Ty = Type::getInt32Ty(Ctx);

  FunctionCallee runtime_wait =
    M->getOrInsertFunction("mdmp_wait", Type::getVoidTy(Ctx), I32Ty);

  FunctionCallee runtime_wait_many =
    M->getOrInsertFunction("mdmp_wait_many",
			   Type::getVoidTy(Ctx),
			   PointerType::getUnqual(Ctx),
			   I32Ty);

  auto Windows = analyseRequestWindows(Requests, RegionEnd, AA, LI, MSSA, M);

  MapVector<Instruction *, SmallVector<const AsyncRequest *, 8>> GroupedWaits;

  for (RequestWindowInfo &Info : Windows) {
    SmallPtrSet<Instruction *, 4> UniqueWaitPoints;

    for (Instruction *RawPt : Info.WaitPoints) {
      Instruction *InsertPt = RawPt;
      
      // if the chosen stop point is exactly mdmp_commregion_end,
      // insert the wait *after* the end call, not before it.
      if (RegionEnd && RawPt == RegionEnd) {
	InsertPt = mdmpInstructionAfter(RawPt);
      }
      
      Instruction *HoistPt = InsertPt;
      Loop *L = LI.getLoopFor(HoistPt->getParent());
      Loop *ReqLoop = LI.getLoopFor(Info.Req->StartPoint->getParent());

      while (L && L != ReqLoop) {
        BasicBlock *HoistTarget = L->getLoopPreheader();
        
        // If the loop isn't simplified yet (no preheader), reliably fall back 
        // to the Immediate Dominator of the loop header
        if (!HoistTarget) {
          if (auto *DomNode = DT.getNode(L->getHeader())->getIDom()) {
            HoistTarget = DomNode->getBlock();
          }
        }

        if (HoistTarget) {
          Instruction *PotentialHoistPt = HoistTarget->getTerminator();
          // Verify we don't accidentally hoist the wait above the async request
          if (DT.dominates(Info.Req->StartPoint, PotentialHoistPt)) {
            HoistPt = PotentialHoistPt;
            L = LI.getLoopFor(HoistPt->getParent());
            continue;
          }
        }
        break; // Stop if we can't safely hoist any further
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
    Function *F = Requests.front().StartPoint->getFunction();
    BasicBlock &EntryBB = F->getEntryBlock();
    IRBuilder<> EntryBuilder(&*EntryBB.getFirstInsertionPt());

    WaitIDsScratchTy = ArrayType::get(I32Ty, MaxBatchSize);
    WaitIDsScratch =
      EntryBuilder.CreateAlloca(WaitIDsScratchTy, nullptr, "mdmp_wait_ids_scratch");
  }

  for (auto &KV : GroupedWaits) {
    Instruction *InsertPt = KV.first;

    // Dedup by token source.
    SmallVector<const AsyncRequest *, 8> UniqueReqs;
    SmallPtrSet<Value *, 8> SeenTokenSources;

    for (const AsyncRequest *Req : KV.second) {
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

}

void MDMPPass::injectThrottledProgress(ArrayRef<AsyncRequest> Requests,
                                       Function &F,
                                       AAResults &AA,
                                       DominatorTree &DT,
                                       LoopInfo &LI,
                                       MemorySSA &MSSA,
                                       Module *M) {
  if (Requests.empty())
    return;

  bool ProgressDebug = mdmpEnvFlagEnabled("MDMP_PROGRESS_DEBUG", false);
  bool UseRelaxedProgressFallback = mdmpEnvFlagEnabled("MDMP_PROGRESS_RELAXED", true);
  unsigned Period = mdmpEnvUnsigned("MDMP_PROGRESS_PERIOD", 64);
  unsigned MaxCallsiteFallbacks = mdmpEnvUnsigned("MDMP_PROGRESS_MAX_CALLSITES", 8);
  unsigned MaxDeepLoopFallbacks = mdmpEnvUnsigned("MDMP_PROGRESS_MAX_DEEP_LOOPS", 2);

  // Profitability guard for aggressive fallback modes only.
  unsigned AggressiveMinReqs = mdmpEnvUnsigned("MDMP_PROGRESS_AGGR_MIN_REQS", 8);
  unsigned AggressiveMinBytes = mdmpEnvUnsigned("MDMP_PROGRESS_AGGR_MIN_BYTES", 131072);
  unsigned AggressiveUnknownMinReqs = mdmpEnvUnsigned("MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS", 8);

  auto isPureComputeLoop = [](Loop *L) {
    for (BasicBlock *BB : L->getBlocksVector()) {
      for (Instruction &I : *BB) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          // Intrinsics and explicitly pure functions are fine
          if (isa<IntrinsicInst>(CB)) continue;
          if (CB->doesNotAccessMemory() || CB->onlyReadsMemory()) continue;

          if (Function *F = CB->getCalledFunction()) {
            StringRef Name = F->getName();
            if (Name.contains("vector") || Name.contains("St6vector")) continue;
          } else {
            // If the function has no name, it's an indirect call (domain.*src).
            // We know these array accessors are pure compute.
            if (CB->isIndirectCall()) {
              if (CB->getType()->isPointerTy() && CB->arg_size() == 2) {
                if (CB->getArgOperand(0)->getType()->isPointerTy() && 
                    CB->getArgOperand(1)->getType()->isIntegerTy()) {
                  continue; // Safe. Do not break the shield.
                }
              }
            }
          }

          // If it survived all checks, it's a real side-effecting function. Break shield.
          return false; 
        }
      }
    }
    return true;
  };
  
  LLVMContext &Ctx = M->getContext();
  IntegerType *I32Ty = Type::getInt32Ty(Ctx);

  FunctionCallee runtime_maybe_progress_site =
    M->getOrInsertFunction("mdmp_maybe_progress_site",
			   Type::getVoidTy(Ctx), I32Ty);

  auto Windows = analyseRequestWindows(Requests, nullptr, AA, LI, MSSA, M);
  if (Windows.empty()) {
    if (ProgressDebug) {
      errs() << "[MDMP PROGRESS] Function " << F.getName()
             << ": no request windows\n";
    }
    return;
  }

  uint64_t TotalPreciseBytes = 0;
  bool HasUnknownBytes = false;

  auto AddBytesSaturating = [&](uint64_t Bytes) {
    if (UINT64_MAX - TotalPreciseBytes < Bytes)
      TotalPreciseBytes = UINT64_MAX;
    else
      TotalPreciseBytes += Bytes;
  };

  for (const AsyncRequest &Req : Requests) {
    for (const TrackedBuffer &Buf : Req.Buffers) {
      if (auto Sz = getPreciseSizeBytes(Buf.Loc.Size)) {
        AddBytesSaturating(*Sz);
      } else {
        HasUnknownBytes = true;
      }
    }
  }

  bool AllowAggressiveFallback = false;

  if (Requests.size() >= AggressiveMinReqs) {
    AllowAggressiveFallback = true;
  } else if (TotalPreciseBytes >= AggressiveMinBytes) {
    AllowAggressiveFallback = true;
  } else if (HasUnknownBytes && Requests.size() >= AggressiveUnknownMinReqs) {
    AllowAggressiveFallback = true;
  }


  SmallVector<Loop *, 16> LeafLoops;
  collectLeafLoops(LI, LeafLoops);

  SmallVector<Loop *, 16> NonLeafLoops;
  collectNonLeafLoops(LI, NonLeafLoops);

  ConstantInt *Zero = ConstantInt::get(I32Ty, 0);
  ConstantInt *One  = ConstantInt::get(I32Ty, 1);

  BasicBlock &EntryBB = F.getEntryBlock();
  Instruction *EntryIP = &*EntryBB.getFirstInsertionPt();
  IRBuilder<> EntryBuilder(EntryIP);

  AllocaInst *CounterAlloc =
    EntryBuilder.CreateAlloca(I32Ty, nullptr, "mdmp_progress_counter");
  EntryBuilder.CreateStore(Zero, CounterAlloc);

  auto InsertThrottledProgressBefore = [&](Instruction *InsertPt, StringRef Kind) {
    unsigned SiteID = NextProgressSiteID++;

    IRBuilder<> Builder(InsertPt);

    Value *Cur = Builder.CreateLoad(I32Ty, CounterAlloc, "mdmp_prog_cur");
    Value *Next = Builder.CreateAdd(Cur, One, "mdmp_prog_next");
    Builder.CreateStore(Next, CounterAlloc);

    Value *DoTick = nullptr;
    if ((Period & (Period - 1)) == 0) {
      Value *Mask = ConstantInt::get(I32Ty, Period - 1);
      Value *Masked = Builder.CreateAnd(Next, Mask, "mdmp_prog_masked");
      DoTick = Builder.CreateICmpEQ(Masked, Zero, "mdmp_prog_tick");
    } else {
      Value *PeriodV = ConstantInt::get(I32Ty, Period);
      Value *Rem = Builder.CreateURem(Next, PeriodV, "mdmp_prog_rem");
      DoTick = Builder.CreateICmpEQ(Rem, Zero, "mdmp_prog_tick");
    }

    Instruction *ThenTerm = SplitBlockAndInsertIfThen(DoTick, InsertPt, false);
    IRBuilder<> ThenBuilder(ThenTerm);
    ThenBuilder.CreateCall(runtime_maybe_progress_site,
                           {ConstantInt::get(I32Ty, SiteID)});

    if (ProgressDebug) {
      errs() << "[MDMP PROGRESS] Inserted " << Kind
             << " progress site in function " << F.getName()
             << " site_id=" << SiteID;
      if (InsertPt->getParent()->hasName())
        errs() << " in block " << InsertPt->getParent()->getName();
      errs() << "\n";
    }
  }; 

  auto InsertDirectProgressBefore = [&](Instruction *InsertPt, StringRef Kind) {
    unsigned SiteID = NextProgressSiteID++;

    IRBuilder<> Builder(InsertPt);
    Builder.CreateCall(runtime_maybe_progress_site,
                       {ConstantInt::get(I32Ty, SiteID)});

    if (ProgressDebug) {
      errs() << "[MDMP PROGRESS] Inserted " << Kind
             << " progress site in function " << F.getName()
             << " site_id=" << SiteID;
      if (InsertPt->getParent()->hasName())
        errs() << " in block " << InsertPt->getParent()->getName();
      errs() << "\n";
    }
  };

  auto LoopMatchesExact = [&](Loop *L) {
    for (const RequestWindowInfo &Info : Windows) {
      if (requestWindowCoversLoopHeader(Info, L, DT))
        return true;
    }
    return false;
  };

  auto LoopMatchesRelaxed = [&](Loop *L) {
    for (const RequestWindowInfo &Info : Windows) {
      if (requestWindowSuggestsLoopProgressRelaxed(Info, L, DT))
        return true;
    }
    return false;
  };

  bool HasExactLeafCandidate = false;
  for (Loop *L : LeafLoops) {
    if (LoopMatchesExact(L)) {
      HasExactLeafCandidate = true;
      break;
    }
  }

  bool UseRelaxedLeafFallback =
    UseRelaxedProgressFallback && !HasExactLeafCandidate;

  if (ProgressDebug) {
    errs() << "[MDMP PROGRESS] Function " << F.getName()
           << ": requests=" << Requests.size()
           << ", windows=" << Windows.size()
           << ", leaf_loops=" << LeafLoops.size()
           << ", nonleaf_loops=" << NonLeafLoops.size()
           << ", exact_leaf_candidates=" << (HasExactLeafCandidate ? "yes" : "no")
           << ", relaxed_fallback=" << (UseRelaxedLeafFallback ? "on" : "off")
           << ", period=" << Period
           << ", max_callsite_fallbacks=" << MaxCallsiteFallbacks
           << ", max_deep_loop_fallbacks=" << MaxDeepLoopFallbacks
           << ", precise_bytes=" << TotalPreciseBytes
           << ", unknown_bytes=" << (HasUnknownBytes ? "yes" : "no")
           << ", aggr_min_reqs=" << AggressiveMinReqs
           << ", aggr_min_bytes=" << AggressiveMinBytes
           << ", aggr_unknown_min_reqs=" << AggressiveUnknownMinReqs
           << ", aggressive_fallback=" << (AllowAggressiveFallback ? "on" : "off")
           << "\n";
  }

  SmallPtrSet<BasicBlock *, 32> InstrumentedHeaders;


  unsigned NumExactLeafInserted = 0;
  unsigned NumRelaxedLeafInserted = 0;
  unsigned NumExactOuterInserted = 0;
  unsigned NumRelaxedOuterInserted = 0;
  unsigned NumCallsiteInserted = 0;
  unsigned NumForcedDeepLeafInserted = 0;

  // ------------------------------------------------------------
  // Stage 1: leaf-loop progress sites
  // ------------------------------------------------------------
  for (Loop *L : LeafLoops) {
    if (isPureComputeLoop(L)) {
      continue; 
    }
    BasicBlock *Header = L->getHeader();
    if (InstrumentedHeaders.contains(Header))
      continue;

    bool ExactMatch = LoopMatchesExact(L);
    bool RelaxedMatch = false;

    if (!ExactMatch && UseRelaxedLeafFallback)
      RelaxedMatch = LoopMatchesRelaxed(L);

    if (!ExactMatch && !RelaxedMatch)
      continue;

    InstrumentedHeaders.insert(Header);
    Instruction *InsertPt = &*Header->getFirstInsertionPt();
    InsertThrottledProgressBefore(InsertPt, ExactMatch ? "exact-leaf" : "relaxed-leaf");

    if (ExactMatch)
      ++NumExactLeafInserted;
    else
      ++NumRelaxedLeafInserted;

    if (ProgressDebug) {
      errs() << "[MDMP PROGRESS] Inserted "
             << (ExactMatch ? "exact-leaf" : "relaxed-leaf")
             << " progress site in function " << F.getName();
      if (Header->hasName())
        errs() << " at loop header " << Header->getName();
      errs() << "\n";
    }
  }

  // ------------------------------------------------------------
  // Stage 2: outer-loop fallback if no leaf-loop site was inserted
  // ------------------------------------------------------------
  if ((NumExactLeafInserted + NumRelaxedLeafInserted) == 0) {
    for (Loop *L : NonLeafLoops) {
      BasicBlock *Header = L->getHeader();
      if (InstrumentedHeaders.contains(Header))
        continue;

      bool ExactMatch = LoopMatchesExact(L);
      bool RelaxedMatch = false;

      if (!ExactMatch && UseRelaxedProgressFallback)
        RelaxedMatch = LoopMatchesRelaxed(L);

      if (!ExactMatch && !RelaxedMatch)
        continue;

      InstrumentedHeaders.insert(Header);
      Instruction *InsertPt = &*Header->getFirstInsertionPt();
      InsertThrottledProgressBefore(InsertPt, ExactMatch ? "exact-outer" : "relaxed-outer");

      if (ExactMatch)
        ++NumExactOuterInserted;
      else
        ++NumRelaxedOuterInserted;

      if (ProgressDebug) {
        errs() << "[MDMP PROGRESS] Inserted "
               << (ExactMatch ? "exact-outer" : "relaxed-outer")
               << " progress site in function " << F.getName();
        if (Header->hasName())
          errs() << " at loop header " << Header->getName();
        errs() << "\n";
      }
    }
  }

  // ------------------------------------------------------------
  // Stage 3: scored callsite fallback if no loop-based site was inserted
  // ------------------------------------------------------------
  if (AllowAggressiveFallback &&
      (NumExactLeafInserted + NumRelaxedLeafInserted +
       NumExactOuterInserted + NumRelaxedOuterInserted) == 0) {

    struct CallsiteCandidate {
      Instruction *Inst = nullptr;
      unsigned Score = 0;
      unsigned LoopDepth = 0;
      bool InLiveBlock = false;
      bool DominatesWait = false;
    };

    SmallVector<CallsiteCandidate, 32> InLoopCandidates;
    SmallVector<CallsiteCandidate, 32> OutOfLoopCandidates;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Instruction *CallI = &I;
        if (!isCandidateCallForProgress(CallI))
          continue;

        bool Match = false;
        unsigned BestScore = 0;
        bool BestInLiveBlock = false;
        bool BestDominatesWait = false;
        unsigned LoopDepth = LI.getLoopDepth(CallI->getParent());

        for (const RequestWindowInfo &Info : Windows) {
          if (!requestWindowSuggestsCallSiteProgressRelaxed(Info, CallI, DT))
            continue;

          Match = true;

          bool InLiveBlock = Info.LiveBlocks.contains(CallI->getParent());
          bool DominatesSomeWait = false;

          for (Instruction *WP : Info.WaitPoints) {
            if (WP->getParent() == CallI->getParent()) {
              if (CallI->comesBefore(WP)) {
                DominatesSomeWait = true;
                break;
              }
            }
            if (DT.dominates(CallI, WP)) {
              DominatesSomeWait = true;
              break;
            }
          }

          unsigned Score = 0;
          if (InLiveBlock)
            Score += 100;
          if (DominatesSomeWait)
            Score += 20;
          if (LoopDepth > 0)
            Score += 50;
          Score += 10 * LoopDepth;

          if (Score > BestScore) {
            BestScore = Score;
            BestInLiveBlock = InLiveBlock;
            BestDominatesWait = DominatesSomeWait;
          }
        }

        if (!Match)
          continue;

        CallsiteCandidate Cand{CallI, BestScore, LoopDepth,
	  BestInLiveBlock, BestDominatesWait};

        if (LoopDepth > 0)
          InLoopCandidates.push_back(Cand);
        else
          OutOfLoopCandidates.push_back(Cand);
      }
    }

    auto SortCandidates = [](SmallVectorImpl<CallsiteCandidate> &Candidates) {
      std::stable_sort(Candidates.begin(), Candidates.end(),
                       [](const CallsiteCandidate &A, const CallsiteCandidate &B) {
                         if (A.Score != B.Score)
                           return A.Score > B.Score;
                         if (A.LoopDepth != B.LoopDepth)
                           return A.LoopDepth > B.LoopDepth;
                         return false;
                       });
    };

    SortCandidates(InLoopCandidates);
    SortCandidates(OutOfLoopCandidates);

    SmallPtrSet<BasicBlock *, 16> CallsiteBlocksUsed;

    auto EmitCandidates = [&](ArrayRef<CallsiteCandidate> Candidates) {
      for (const CallsiteCandidate &Cand : Candidates) {
        Instruction *CallI = Cand.Inst;

        // Avoid spamming one block with multiple fallback sites.
        if (!CallsiteBlocksUsed.insert(CallI->getParent()).second)
          continue;

        // Callsite fallback is intentionally unthrottled. These sites are sparse
        // and are the only practical progress points in many irregular kernels.
        InsertDirectProgressBefore(CallI, "callsite");
        ++NumCallsiteInserted;

        if (ProgressDebug) {
          errs() << "[MDMP PROGRESS] Inserted callsite progress site in function "
                 << F.getName();

          if (auto *CB = dyn_cast<CallBase>(CallI)) {
            if (Function *Callee = CB->getCalledFunction())
              errs() << " before call " << Callee->getName();
            else
              errs() << " before indirect call";
          }

          errs() << " loop_depth=" << Cand.LoopDepth
                 << " live_block=" << (Cand.InLiveBlock ? "yes" : "no")
                 << " dominates_wait=" << (Cand.DominatesWait ? "yes" : "no")
                 << " score=" << Cand.Score
                 << "\n";
        }

        if (NumCallsiteInserted >= MaxCallsiteFallbacks)
          return;
      }
    };

    // First try only in-loop callsites.
    EmitCandidates(InLoopCandidates);

    // If we still found nothing useful, fall back to out-of-loop callsites.
    if (NumCallsiteInserted == 0) {
      EmitCandidates(OutOfLoopCandidates);
    }
  }

  // ------------------------------------------------------------
  // Stage 4: last-chance deep-leaf-loop fallback
  //
  // If we still failed to place any progress site at all, force a small
  // number of throttled progress sites into the deepest leaf loops whose
  // headers are dominated by at least one request start.
  // ------------------------------------------------------------
  if (AllowAggressiveFallback &&
      MaxDeepLoopFallbacks > 0 &&
      (NumExactLeafInserted + NumRelaxedLeafInserted +
       NumExactOuterInserted + NumRelaxedOuterInserted +
       NumCallsiteInserted) == 0) {
    
    struct DeepLoopCandidate {
      Loop *L = nullptr;
      unsigned Depth = 0;
      unsigned NumBlocks = 0;
    };

    SmallVector<DeepLoopCandidate, 16> DeepCandidates;

    for (Loop *L : LeafLoops) {

      if (isPureComputeLoop(L)) {
	continue; 
      }

      BasicBlock *Header = L->getHeader();
      if (InstrumentedHeaders.contains(Header))
	continue;

      bool MatchesSomeLiveWindow = false;

      for (const RequestWindowInfo &Info : Windows) {
	for (BasicBlock *LoopBB : L->getBlocksVector()) {
	  if (Info.LiveBlocks.contains(LoopBB)) {
	    MatchesSomeLiveWindow = true;
	    break;
	  }
	}

	if (MatchesSomeLiveWindow)
	  break;

	for (Instruction *WP : Info.WaitPoints) {
	  if (L->contains(WP->getParent())) {
	    MatchesSomeLiveWindow = true;
	    break;
	  }
	}

	if (MatchesSomeLiveWindow)
	  break;
      }

      if (!MatchesSomeLiveWindow)
	continue;

      DeepLoopCandidate Cand;
      Cand.L = L;
      Cand.Depth = LI.getLoopDepth(Header);
      Cand.NumBlocks = (unsigned)L->getBlocksVector().size();
      DeepCandidates.push_back(Cand);
    }   

    std::stable_sort(DeepCandidates.begin(), DeepCandidates.end(),
                     [](const DeepLoopCandidate &A,
                        const DeepLoopCandidate &B) {
                       if (A.Depth != B.Depth)
                         return A.Depth > B.Depth;
                       if (A.NumBlocks != B.NumBlocks)
                         return A.NumBlocks > B.NumBlocks;
                       return false;
                     });

    for (const DeepLoopCandidate &Cand : DeepCandidates) {
      BasicBlock *Header = Cand.L->getHeader();
      if (!InstrumentedHeaders.insert(Header).second)
        continue;

      Instruction *InsertPt = &*Header->getFirstInsertionPt();
      InsertThrottledProgressBefore(InsertPt, "forced-deep-leaf");
      ++NumForcedDeepLeafInserted;

      if (ProgressDebug) {
        errs() << "[MDMP PROGRESS] Inserted forced-deep-leaf progress site in function "
               << F.getName();
        if (Header->hasName())
          errs() << " at loop header " << Header->getName();
        errs() << " loop_depth=" << Cand.Depth
               << " loop_blocks=" << Cand.NumBlocks
               << "\n";
      }

      if (NumForcedDeepLeafInserted >= MaxDeepLoopFallbacks)
        break;
    }
  }


  if (ProgressDebug) {
    errs() << "[MDMP PROGRESS] Function " << F.getName()
           << ": inserted exact-leaf=" << NumExactLeafInserted
           << ", relaxed-leaf=" << NumRelaxedLeafInserted
           << ", exact-outer=" << NumExactOuterInserted
           << ", relaxed-outer=" << NumRelaxedOuterInserted
           << ", callsite=" << NumCallsiteInserted
           << ", forced-deep-leaf=" << NumForcedDeepLeafInserted
           << "\n";
  }

}
static void addMDMPWithCleanupPipeline(ModulePassManager &MPM) {
  MPM.addPass(MDMPPass());

  FunctionPassManager FPM;
#if LLVM_VERSION_GE(22, 0)
  FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
#else
  FPM.addPass(SROAPass());
#endif
  FPM.addPass(EarlyCSEPass());
  FPM.addPass(InstCombinePass());
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(InstCombinePass());
  FPM.addPass(DCEPass());

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.addPass(GlobalDCEPass());
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "MDMP", "v0.5",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
					 [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
					   if (Name == "mdmp") {
					     addMDMPWithCleanupPipeline(MPM);
					     return true;
					   }
					   return false;
					 });

      // Standard compilation: Run as early as possible, but skip PreLink 
      // so our programmatic inliner has access to all merged files.
      PB.registerOptimizerEarlyEPCallback(
					  [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
					    if (Phase == ThinOrFullLTOPhase::ThinLTOPreLink || 
						Phase == ThinOrFullLTOPhase::FullLTOPreLink) {
					      return; 
					    }
					    addMDMPWithCleanupPipeline(MPM);
					  });

      // LTO Link Phase: Run early, before SimplifyCFG does tail merging, which can cause MDMP functions
      // to be moved incorrectly.
      PB.registerFullLinkTimeOptimizationEarlyEPCallback(
							 [](ModulePassManager &MPM, OptimizationLevel Level) {
							   addMDMPWithCleanupPipeline(MPM);
							 });
    }
  };
}
