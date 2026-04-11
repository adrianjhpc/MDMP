#ifndef MDMP_PASS_H
#define MDMP_PASS_H

#define LLVM_VERSION_GE(major, minor)					\
  (LLVM_VERSION_MAJOR > (major) ||					\
   LLVM_VERSION_MAJOR == (major) && LLVM_VERSION_MINOR >= (minor))

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"

#if LLVM_VERSION_GE(23, 0)
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include <vector>
#include <optional>
#include <cassert>
#include <limits>
#include <cstdlib>
#include <cstring>

namespace llvm {

  class MDMPPass : public PassInfoMixin<MDMPPass> {
  public:
    // Main entry point for the Pass Manager
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

    struct TrackedBuffer {
      MemoryLocation Loc;
      bool isNetworkReadOnly; // True = Send buffer (CPU reads ok), False = Recv buffer (CPU reads/writes collide)
    };

    struct AsyncRequest {
      llvm::Value *WaitTokenValue = nullptr;      // Direct SSA token when usable
      llvm::AllocaInst *WaitTokenAlloc = nullptr; // Spill-slot fallback
      llvm::CallInst *StartPoint = nullptr;       // Where the request goes in flight
      std::vector<TrackedBuffer> Buffers;
    };

    struct CompletedRegion {
      llvm::Instruction *RegionEnd = nullptr;
      std::vector<AsyncRequest> Requests;
    };
    
    struct TraversalState {
      BasicBlock *BB;
      BasicBlock::iterator StartIt;
    };

    struct RequestWindowInfo {
      const AsyncRequest *Req = nullptr;
      SmallVector<Instruction *, 4> WaitPoints;
      SmallPtrSet<BasicBlock *, 32> LiveBlocks;
    };        
    
  private:

    unsigned NextProgressSiteID = 0;   
    
    std::vector<CompletedRegion> CompletedRegions;    

    bool isHardBarrierInstForWaitPlacement(Instruction *Inst);

    bool isAsyncMDMPInstForWaitPlacement(Instruction *Inst);

    Instruction *mdmpInstructionAfter(Instruction *I);
    
    bool instructionIsTrueConsumerOrClobber(Instruction *I, const TrackedBuffer &Buf, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);

    bool instructionTouchesAnyTrackedBufferPhase2(Instruction *I, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);

    Instruction *findFirstTrueConflictInBlock(BasicBlock *BB, BasicBlock::iterator StartIt, Instruction *RegionEnd, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);
    
    bool isIgnorableIntrinsicForMDMP(Instruction *I);

    bool instructionConflictsWithTrackedBufferMSSA(Instruction *I, const TrackedBuffer &Buf, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);

    bool instructionConflictsWithAnyTrackedBufferMSSA(Instruction *I, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);
    
    bool waitTokenValueDominates(Value *V, Instruction *InsertPt, DominatorTree &DT);
    
    Value *materialiseWaitTokenForUse(const AsyncRequest &Req, Instruction *InsertPt, IntegerType *I32Ty, IRBuilder<> &Builder, DominatorTree &DT);
    
    bool isAsyncMDMPOpName(StringRef FnName);

    bool isHardBarrierCallName(StringRef Name);

    void collectLeafLoops(Loop *L, SmallVectorImpl<Loop *> &Out);

    void collectLeafLoops(LoopInfo &LI, SmallVectorImpl<Loop *> &Out);

    void collectNonLeafLoops(Loop *L, SmallVectorImpl<Loop *> &Out);

    void collectNonLeafLoops(LoopInfo &LI, SmallVectorImpl<Loop *> &Out);

    bool requestWindowSuggestsCallSiteProgressRelaxed(const RequestWindowInfo &Info, Instruction *Inst, DominatorTree &DT);

    bool isCandidateCallForProgress(Instruction *Inst);

    bool isClearlyUnhelpfulProgressCallName(StringRef Name);

    bool requestWindowCoversLoopHeader(const RequestWindowInfo &Info, Loop *L, DominatorTree &DT);

    bool requestWindowSuggestsLoopProgressRelaxed(const RequestWindowInfo &Info, Loop *L, DominatorTree &DT);

    bool loopMayConflictWithTrackedBuffers(Loop *L, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);

    bool shouldForceWaitAtLoopBackedge(Loop *EdgeLoop, Loop *ReqLoop, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, MemorySSA &MSSA, const DataLayout &DL);
    
    SmallVector<RequestWindowInfo, 8> analyseRequestWindows(ArrayRef<AsyncRequest> Requests, Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, MemorySSA &MSSA, Module *M);
    
    // Global tracker for the current function being processed
    std::vector<AsyncRequest> PendingRequests;
  
    std::optional<uint64_t> getConstU64(Value *V);

    std::optional<uint64_t> getStaticMPITypeBytes(Value *TypeCodeV);

    std::optional<uint64_t> checkedMulU64(uint64_t A, uint64_t B);

    LocationSize derivePreciseSpan(Value *CountV, Value *TypeCodeV, Value *BytesV);

    TrackedBuffer makePreciseTrackedBuffer(Value *Ptr, Value *CountV, Value *TypeCodeV, Value *BytesV, bool IsNetworkReadOnly);

    TrackedBuffer makeUnknownTrackedBuffer(Value *Ptr, bool IsNetworkReadOnly);

    std::optional<uint64_t> getPreciseSizeBytes(LocationSize S);

    bool areDefinitelyDisjoint(const MemoryLocation &A, const MemoryLocation &B, const DataLayout &DL);

    bool locationsMayOverlap(const MemoryLocation &A, const MemoryLocation &B, AAResults &AA, const DataLayout &DL);
    
    bool isHardMotionBarrier(Instruction *I);
    
    bool operandsAvailableBefore(CallInst *CI, Instruction *InsertBefore, DominatorTree &DT);

    bool instructionConflictsWithTrackedBuffer(Instruction *I, const TrackedBuffer &Buf, AAResults &AA, const DataLayout &DL);

    bool instructionConflictsWithAnyTrackedBuffer(Instruction *I, ArrayRef<TrackedBuffer> Buffers, AAResults &AA, const DataLayout &DL);
    
    BasicBlock *getLinearPredecessor(BasicBlock *BB);

    std::vector<TrackedBuffer> buildSendRecvBuffers(Value *Buf, Value *Count, Value *Type, Value *Bytes, bool IsSend);

    std::vector<TrackedBuffer> buildReduceBuffers(Value *SendBuf, Value *RecvBuf, Value *Count, Value *Type, Value *Bytes);

    std::vector<TrackedBuffer> buildGatherBuffers(Value *SendBuf, Value *SendCount, Value *RecvBuf, Value *Type, Value *Bytes);

    std::vector<TrackedBuffer> buildAllreduceBuffers(Value *SendBuf, Value *RecvBuf, Value *Count, Value *Type, Value *Bytes);

    std::vector<TrackedBuffer> buildAllgatherBuffers(Value *SendBuf, Value *Count, Value *RecvBuf, Value *Type, Value *Bytes);

    std::vector<TrackedBuffer> buildBcastBuffers(Value *Buf, Value *Count, Value *Type, Value *Bytes);

    bool inlineThinMDMPWrappers(Module &M);
    
    // Core function processor requiring Alias, Dominator, and Loop analyses
    bool runOnFunction(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI);
		       

    // Translates user markers to either Imperative (Send/Recv) or Declarative (Commit) calls
    bool transformFunctionsToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI);
				   

    // Accepts a vector of MemoryLocations to safely hoist 
    // collectives and declarative bulk-commits without breaking data dependencies.
    void hoistInitiation(CallInst *CI, std::vector<TrackedBuffer> &Locs, AAResults &AA, DominatorTree &DT);

    // FG Traversal Engine: Uses LoopInfo to prevent "Inner-Loop Poisoning" 
    // by ensuring waits are safely dropped at the loop preheader/terminator boundaries.    
    void injectWaitsForRegion(ArrayRef<AsyncRequest> Requests, Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, LLVMContext &Ctx, Module *M, DominatorTree &DT, MemorySSA &MSSA);
    
    void injectThrottledProgress(ArrayRef<AsyncRequest> Requests, Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI, MemorySSA &MSSA, Module *M);

  };

} // namespace llvm

#endif // MDMP_PASS_H
