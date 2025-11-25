#ifndef LLVM_TRANSFORMS_ZLUDA_SPLITMMA_H
#define LLVM_TRANSFORMS_ZLUDA_SPLITMMA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SplitMMAPass : public PassInfoMixin<SplitMMAPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_ZLUDA_SPLITMMA_H
