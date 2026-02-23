#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  struct MDMPManagedPass : public FunctionPass {
    static char ID;
    MDMPManagedPass() : FunctionPass(ID) {}
    
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AAResultsWrapperPass>(); // Required for dependency checking
      AU.setPreservesCFG();
    }
    
    bool runOnFunction(Function &F) override {
      auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      bool modified = false;
      
      for (auto &BB : F) {
	for (auto BI = BB.begin(), BE = BB.end(); BI != BE; ) {
	  Instruction *Inst = &*BI++;
	  
	  if (auto *CI = dyn_cast<CallInst>(Inst)) {
	    Function *Callee = CI->getCalledFunction();
	    if (!Callee) continue;
	    
	    // Hoisting: Move Send/Recv Start calls upward
	    if (Callee->getName() == "__mdmp_initiate_comm") {
	      modified |= hoistInitiation(CI, AA);
	    }
	    
	    // Sinking: Move Wait calls downward
	    if (Callee->getName() == "__mdmp_wait_comm") {
	      modified |= sinkCompletion(CI, AA);
	    }
	  }
	}
      }
      return modified;
    }
    
    
    // Hoists the communication start until it hits a data dependency
    bool hoistInitiation(CallInst *CommCall, AliasAnalysis &AA) {
      Value *Buffer = CommCall->getArgOperand(0); 
      // Use LocationSize::beforeOrAfterPointer() if the exact size isn't known at compile time
      MemoryLocation Loc(Buffer, LocationSize::beforeOrAfterPointer());
      bool moved = false;
      
      BasicBlock::iterator it(CommCall);
      while (it != CommCall->getParent()->begin()) {
	Instruction *Prev = &*(--it);
	
	// Use isModSet to check if the instruction modifies the buffer
	if (isModSet(AA.getModRefInfo(Prev, Loc))) {
	  break; // Data dependency found: stop hoisting
	}
	
	CommCall->moveBefore(Prev);
	moved = true;
      }
      return moved;
    }
    
    // Sinks the wait call until it hits a data dependency
    bool sinkCompletion(CallInst *WaitCall, AliasAnalysis &AA) {
      Value *Buffer = WaitCall->getArgOperand(0);
      MemoryLocation Loc(Buffer, LocationSize::beforeOrAfterPointer());
      bool moved = false;
      
      BasicBlock::iterator it(WaitCall);
      it++; // Start checking after the current WaitCall
      while (it != WaitCall->getParent()->end()) {
	Instruction *Next = &*it++;
	
	// Use isModOrRefSet to check if the instruction either reads or writes the buffer
	if (isModOrRefSet(AA.getModRefInfo(Next, Loc))) {
	  break; // Buffer is used or modified here: stop sinking
	}
	
	WaitCall->moveAfter(Next);
	moved = true;
      }
      return moved;
    }
    
  };
}

char MDMPManagedPass::ID = 0;
static RegisterPass<MDMPManagedPass> Y("mdmp-managed", "Hoist/Sink MDMP Communications for Overlap");
