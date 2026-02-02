// MDMPPragmaPass.h
#ifndef MDMP_PRAGMA_PASS_H
#define MDMP_PRAGMA_PASS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
#include <set>

namespace llvm {
    class MDMPPragmaPass : public ModulePass {
    private:
        // Pragma processing context
        struct PragmaContext {
            bool in_overlap_region = false;
            int current_tag = 0;
            int current_rank = 0;
            bool is_async = false;
            bool is_optimized = true;
            int optimization_level = 2;
        };
        
        // Pragma processing state
        std::map<Function*, PragmaContext> function_contexts;
        std::set<BasicBlock*> processed_blocks;
        
    public:
        static char ID;
        MDMPPragmaPass() : ModulePass(ID) {}
        
        bool runOnModule(Module &M) override;
        bool runOnFunction(Function &F);
        
    private:
        void processPragmaDirectives(Function &F);
        void processOverlapRegion(Function &F);
        void processMemoryManagement(Function &F);
        void processCommunicationPatterns(Function &F);
        void transformPragmasToCalls(Function &F);
        void analyzePragmas(Function &F);
        void insertMDMPRuntimeCalls(Function &F);
        
        // Pragma parsing helpers
        bool parseMDMPPragma(const std::string &pragma_line);
        void handleCommunicationBegin();
        void handleCommunicationEnd();
        void handleSync();
	void handleWait();
        void handleSend();
	void handleRecv();
        void handleOptimize(int level);
        void handleNoOpt();
        void handleRank();
	void handleSize();
        void handleReduce(int op, void* src, void* dst, size_t size);
        void handleBarrier();
        void handleBroadcast(void* data, size_t size, int root);
        void handleGather(void* sendbuf, void* recvbuf, int count, 
                         int datatype, int root);
        void handleScatter(void* sendbuf, void* recvbuf, int count,
                          int datatype, int root);
    };
}

#endif

