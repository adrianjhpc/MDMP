#ifndef MDMP_PRAGMA_PASS_H
#define MDMP_PRAGMA_PASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include <map>
#include <vector>

namespace llvm {
    class MDMPPragmaPass : public PassInfoMixin<MDMPPragmaPass> {
    private:
        struct ActiveRequest {
            MemoryLocation Loc;
            CallInst *RuntimeCall;
        };
        std::vector<ActiveRequest> PendingRequests;


    public:
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

    private:
        bool runOnFunction(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI);
        void transformPragmasToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI);
    
        void hoistInitiation(CallInst *CI, MemoryLocation &Loc, AAResults &AA, DominatorTree &DT, LoopInfo &LI, bool isSend);
        bool sinkCompletion(CallInst *WaitCall, AAResults &AA);
        void injectWaitsForRegion(Instruction *RegionEnd, AAResults &AA, LLVMContext &Ctx, Module *M);
    };
}

#endif
