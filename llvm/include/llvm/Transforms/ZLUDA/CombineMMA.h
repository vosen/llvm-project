#ifndef LLVM_TRANSFORMS_ZLUDA_COMBINEMMA_H
#define LLVM_TRANSFORMS_ZLUDA_COMBINEMMA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class CombineMMAPass : public PassInfoMixin<CombineMMAPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_ZLUDA_COMBINEMMA_H
