#ifndef LLVM_TRANSFORMS_ZLUDA_LOWERMATRIXCONVERSIONS_H
#define LLVM_TRANSFORMS_ZLUDA_LOWERMATRIXCONVERSIONS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LowerMatrixConversionsPass : public PassInfoMixin<LowerMatrixConversionsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_ZLUDA_LOWERMATRIXCONVERSIONS_H
