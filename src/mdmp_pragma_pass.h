#ifndef MDMP_PRAGMA_PASS_H
#define MDMP_PRAGMA_PASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
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
        bool runOnFunction(Function &F, AAResults &AA);
        void transformPragmasToCalls(Function &F, AAResults &AA);
        
        // Hoister & Sinker logic with MemoryLocation awareness
        bool hoistInitiation(CallInst *CommCall, MemoryLocation Loc, AAResults &AA, bool isSend);
        bool sinkCompletion(CallInst *WaitCall, AAResults &AA);
        void injectWaitsForRegion(Instruction *RegionEnd, AAResults &AA, LLVMContext &Ctx, Module *M);
    };
}

#endif
