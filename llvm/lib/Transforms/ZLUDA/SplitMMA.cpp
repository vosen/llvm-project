#include "llvm/Transforms/ZLUDA/SplitMMA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

static IntrinsicInst *getZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA &&
      MMA->getIntrinsicID() == Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    return MMA;
  }
  return nullptr;
}

// Split one NVIDIA-style 16x8x32 MMA instruction into two NVIDIA-style 16x8x16
// MMA instructions.
static bool splitMMA(IntrinsicInst *MMA) {
  assert(MMA->getIntrinsicID() == Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32);

  Value *A = MMA->getArgOperand(0);
  Value *B = MMA->getArgOperand(1);
  Value *C = MMA->getArgOperand(2);

  IRBuilder<> Builder(MMA);

  auto *V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  auto *V2I32Ty = VectorType::get(Builder.getInt32Ty(), 2, /*Scalable=*/false);
  auto *V2I32x2Ty = StructType::get(Builder.getContext(), {V2I32Ty, V2I32Ty});
  auto *V1I32Ty = VectorType::get(Builder.getInt32Ty(), 1, /*Scalable=*/false);
  auto *V1I32x2Ty = StructType::get(Builder.getContext(), {V1I32Ty, V1I32Ty});

  auto *SplitA = Builder.CreateIntrinsic(
      V2I32x2Ty, Intrinsic::zluda_amatrix_split_nv16x32, {A});
  auto *A0 = Builder.CreateExtractValue(SplitA, {0});
  auto *A1 = Builder.CreateExtractValue(SplitA, {1});

  auto *SplitB = Builder.CreateIntrinsic(
      V1I32x2Ty, Intrinsic::zluda_bmatrix_split_nv32x8, {B});
  auto *B0 = Builder.CreateExtractValue(SplitB, {0});
  auto *B1 = Builder.CreateExtractValue(SplitB, {1});

  auto *Accumulator = Builder.CreateIntrinsic(
      V4I32Ty, Intrinsic::zluda_mma_m16n8k16_s32_s8_s8_s32, {A0, B0, C});
  auto *Result = Builder.CreateIntrinsic(
      V4I32Ty, Intrinsic::zluda_mma_m16n8k16_s32_s8_s8_s32,
      {A1, B1, Accumulator});

  MMA->replaceAllUsesWith(Result);
  MMA->eraseFromParent();

  return true;
}

static bool splitBB(BasicBlock &BB) {
  bool Modified = false;

  for (Instruction &I : make_early_inc_range(BB)) {
    auto *MMA = getZludaMMA(I);
    if (!MMA) {
      continue;
    }

    if (splitMMA(MMA)) {
      Modified = true;
    }
  }

  return Modified;
}

static bool split(Function &F) {
  bool Modified = false;

  for (BasicBlock &BB : F) {
    Modified |= splitBB(BB);
  }

  return Modified;
}

PreservedAnalyses SplitMMAPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (split(F)) {
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  }

  return PreservedAnalyses::all();
}
