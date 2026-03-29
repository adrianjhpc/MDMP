#ifndef MDMP_PASS_H
#define MDMP_PASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include <vector>

namespace llvm {

class MDMPPass : public PassInfoMixin<MDMPPass> {
public:
    // Main entry point for the Pass Manager
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

private:
    // Core function processor requiring Alias, Dominator, and Loop analyses
    bool runOnFunction(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI);
    
    // Translates user markers to either Imperative (Send/Recv) or Declarative (Commit) calls
    void transformFunctionsToCalls(Function &F, AAResults &AA, DominatorTree &DT, LoopInfo &LI);
    
    // Upgraded Hoisting Engine: Accepts a vector of MemoryLocations to safely hoist 
    // collectives and declarative bulk-commits without breaking data dependencies.
    void hoistInitiation(CallInst *CI, std::vector<MemoryLocation> &Locs, 
                         AAResults &AA, DominatorTree &DT, LoopInfo &LI, bool isSend);
    
    // FG Traversal Engine: Uses LoopInfo to prevent "Inner-Loop Poisoning" 
    // by ensuring waits are safely dropped at the loop preheader/terminator boundaries.
    void injectWaitsForRegion(Instruction *RegionEnd, AAResults &AA, LoopInfo &LI, 
                              LLVMContext &Ctx, Module *M);
};

} // namespace llvm

#endif // MDMP_PASS_H
