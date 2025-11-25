#include "llvm/Transforms/ZLUDA/CombineMMA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

// Moves the instructions that FromBefore depends on to before ToBefore. Does
// nothing other than return false if FromBefore has a dependency on ToBefore,
// and true otherwise. Based on LoadStoreVectorizer's reorder.
static bool tryToReorderOperands(IntrinsicInst *FromBefore,
                                 IntrinsicInst *ToBefore) {
  assert(FromBefore->getParent() == ToBefore->getParent());

  SmallPtrSet<Instruction *, 16> InstructionsToMove;
  SmallVector<Instruction *, 16> Worklist;

  Worklist.emplace_back(FromBefore);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    for (Value *Operand : I->operands()) {
      auto *Dependency = dyn_cast<Instruction>(Operand);
      if (!Dependency || Dependency->getOpcode() == Instruction::PHI) {
        continue;
      }

      // Ignore instructions outside of the current basic block
      if (Dependency->getParent() != FromBefore->getParent()) {
        continue;
      }

      if (Dependency == ToBefore) {
        return false;
      }

      assert(Dependency != FromBefore &&
             "Unexpected cycle while re-ordering instructions");

      if (!Dependency->comesBefore(ToBefore)) {
        InstructionsToMove.insert(Dependency);
        Worklist.emplace_back(Dependency);
      }
    }
  }

  // We only need to move the instructions in between ToBefore and FromBefore
  for (auto BBI = ToBefore->getIterator(), E = FromBefore->getIterator();
       BBI != E; ++BBI) {
    auto I = &*BBI;
    if (InstructionsToMove.contains(I)) {
      I->moveBefore(ToBefore);
    }
  }

  return true;
}

static IntrinsicInst *getF32ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA && MMA->getIntrinsicID() ==
                 Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    return MMA;
  }
  return nullptr;
}

static IntrinsicInst *getS32ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA &&
      MMA->getIntrinsicID() == Intrinsic::zluda_mma_m16n8k16_s32_s8_s8_s32) {
    return MMA;
  }
  return nullptr;
}

enum class OperandType {
  F32,
  BF16,
  S32,
  S8,
};

static OperandType getABOperandType(Intrinsic::ID IID) {
  switch (IID) {
  case Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32:
    return OperandType::BF16;
  case Intrinsic::zluda_mma_m16n8k16_s32_s8_s8_s32:
    return OperandType::S8;
  default:
    llvm_unreachable("Unsupported MMA intrinsic");
  }
}

static OperandType getCDOperandType(Intrinsic::ID IID) {
  switch (IID) {
  case Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32:
    return OperandType::F32;
  case Intrinsic::zluda_mma_m16n8k16_s32_s8_s8_s32:
    return OperandType::S32;
  default:
    llvm_unreachable("Unsupported MMA intrinsic");
  }
}

class MMACombiner {
public:
  bool combine(Function &F);

private:
  bool combineBB(BasicBlock &BB);
  bool combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs);
  bool combineMMA(IntrinsicInst *First, IntrinsicInst *Second);

  void lowerMMA(IntrinsicInst *MMA);

  Value *combineC(IRBuilder<> &Builder, Value *FirstC, Value *SecondC);
  Value *convertC(IRBuilder<> &Builder, Value *C);

  SmallVector<Instruction *> MaybeRemove;
};

// If FirstC and SecondC are the result of a split, return the value before it
// was split. Otherwise concatenate the matrices.
Value *MMACombiner::combineC(IRBuilder<> &Builder, Value *FirstC,
                             Value *SecondC) {
  auto *FirstExtract = dyn_cast<ExtractValueInst>(FirstC);
  auto *SecondExtract = dyn_cast<ExtractValueInst>(SecondC);
  if (FirstExtract != nullptr && SecondExtract != nullptr) {
    auto *FirstAggregate = FirstExtract->getAggregateOperand();
    auto *SecondAggregate = SecondExtract->getAggregateOperand();
    if (FirstAggregate == SecondAggregate) {
      if (auto *II = dyn_cast<IntrinsicInst>(FirstAggregate)) {
        if (II->getIntrinsicID() ==
            Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16_f32) {
          MaybeRemove.emplace_back(FirstExtract);
          MaybeRemove.emplace_back(SecondExtract);
          MaybeRemove.emplace_back(II);
          return II->getArgOperand(0);
        }
      }
    }
  }

  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);

  return Builder.CreateIntrinsic(
      V8F32Ty, Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8_f32,
      {FirstC, SecondC});
}

// If C is the result of a truncate, return the value before it was truncated.
// Otherwise zero-extend the matrix.
Value *MMACombiner::convertC(IRBuilder<> &Builder, Value *C) {
  if (auto *II = dyn_cast<IntrinsicInst>(C)) {
    if (II->getIntrinsicID() ==
        Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16_f32) {
      MaybeRemove.emplace_back(II);
      return II->getArgOperand(0);
    }
  }

  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);

  return Builder.CreateIntrinsic(
      V8F32Ty, Intrinsic::zluda_cmatrix_zext_amd16x16_nv16x8_f32, {C});
}

// Combine two NVIDIA-style 16x8 MMA instructions into one AMD-style 16x16 MMA
// instruction.
bool MMACombiner::combineMMA(IntrinsicInst *First, IntrinsicInst *Second) {
  assert(First->getIntrinsicID() ==
             Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32 &&
         Second->getIntrinsicID() ==
             Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32);
  Value *FirstA = First->getArgOperand(0);
  Value *FirstB = First->getArgOperand(1);
  Value *FirstC = First->getArgOperand(2);

  Value *SecondA = Second->getArgOperand(0);
  Value *SecondB = Second->getArgOperand(1);
  Value *SecondC = Second->getArgOperand(2);

  if (FirstA != SecondA) {
    return false;
  }

  // We try to move all operands of Second before First. If we cannot, it is
  // because Second has a dependency on first, and we cannot combine them.
  if (!tryToReorderOperands(Second, First)) {
    return false;
  }

  // We insert before the first MMA, in case it has any users before the second
  // MMA. Any dependencies of the second MMA that come after the first MMA will
  // be reordered later.
  IRBuilder<> Builder(First);

  auto V4F32Ty = VectorType::get(Builder.getFloatTy(), 4, /*Scalable=*/false);
  auto V4F32x2Ty = StructType::get(Builder.getContext(), {V4F32Ty, V4F32Ty});
  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);
  auto V16I16Ty = VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false);

  auto ShuffledA = Builder.CreateIntrinsic(
      V16I16Ty, Intrinsic::zluda_amatrix_convert_amd_nv16x16_bf16, {FirstA});
  auto CombinedB = Builder.CreateIntrinsic(
      V16I16Ty, Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8_bf16,
      {FirstB, SecondB});
  auto CombinedC = combineC(Builder, FirstC, SecondC);

  auto *Result =
      Builder.CreateIntrinsic(V8F32Ty, Intrinsic::amdgcn_wmma_f32_16x16x16_bf16,
                              {ShuffledA, CombinedB, CombinedC});
  auto *Split = Builder.CreateIntrinsic(
      V4F32x2Ty, Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16_f32, {Result});

  auto NewFirst = Builder.CreateExtractValue(Split, {0});
  auto NewSecond = Builder.CreateExtractValue(Split, {1});

  First->replaceAllUsesWith(NewFirst);
  Second->replaceAllUsesWith(NewSecond);

  First->eraseFromParent();
  Second->eraseFromParent();

  return true;
}

// Lower an NVIDIA-style 16x8 MMA instruction to an AMD-style 16x16 MMA
// instruction. The unused part of the matrix is filled with zeroes.
void MMACombiner::lowerMMA(IntrinsicInst *MMA) {
  assert(MMA->getIntrinsicID() ==
         Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32);

  Value *A = MMA->getArgOperand(0);
  Value *B = MMA->getArgOperand(1);
  Value *C = MMA->getArgOperand(2);

  IRBuilder<> Builder(MMA);

  auto V4F32Ty = VectorType::get(Builder.getFloatTy(), 4, /*Scalable=*/false);
  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);
  auto V16I16Ty = VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false);

  auto ShuffledA = Builder.CreateIntrinsic(
      V16I16Ty, Intrinsic::zluda_amatrix_convert_amd_nv16x16_bf16, {A});
  auto ShuffledB = Builder.CreateIntrinsic(
      V16I16Ty, Intrinsic::zluda_bmatrix_zext_amd16x16_nv16x8_bf16, {B});
  auto ShuffledC = convertC(Builder, C);

  auto *Result =
      Builder.CreateIntrinsic(V8F32Ty, Intrinsic::amdgcn_wmma_f32_16x16x16_bf16,
                              {ShuffledA, ShuffledB, ShuffledC});
  auto *Truncated = Builder.CreateIntrinsic(
      V4F32Ty, Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16_f32, {Result});

  MMA->replaceAllUsesWith(Truncated);
  MMA->eraseFromParent();
}

bool MMACombiner::combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs) {
  bool Modified = false;

  IntrinsicInst *PrevMMA = nullptr;

  for (IntrinsicInst *MMA : MMAs) {
    if (PrevMMA != nullptr) {
      // Try to combine PrevMMA with current MMA.
      if (combineMMA(PrevMMA, MMA)) {
        Modified = true;
        PrevMMA = nullptr;
        continue;
      }

      // Cannot combine, so lower PrevMMA individually.
      lowerMMA(PrevMMA);
      Modified = true;
    }
    PrevMMA = MMA;
  }

  if (PrevMMA) {
    lowerMMA(PrevMMA);
    Modified = true;
  }

  return Modified;
}

bool MMACombiner::combineBB(BasicBlock &BB) {
  // For now, we simply combine adjacent m16n8k16 MMAs if possible. This may be
  // good enough in most cases. Any MMAs that cannot be combined are lowered
  // individually.
  bool Modified = false;

  SmallVector<IntrinsicInst *> F32MMAs;
  SmallVector<IntrinsicInst *> S32MMAs;

  for (Instruction &I : BB) {
    auto *F32MMA = getF32ZludaMMA(I);
    if (F32MMA) {
      F32MMAs.push_back(F32MMA);
    }
    auto *S32MMA = getS32ZludaMMA(I);
    if (S32MMA) {
      S32MMAs.push_back(S32MMA);
    }
  }

  Modified |= combineMMAs(F32MMAs);
  Modified |= combineMMAs(S32MMAs);

  return Modified;
}

bool MMACombiner::combine(Function &F) {
  bool Modified = false;

  for (BasicBlock &BB : F) {
    Modified |= combineBB(BB);
  }

  for (Instruction *I : MaybeRemove) {
    if (I->user_empty()) {
      I->eraseFromParent();
      Modified = true;
    }
  }

  return Modified;
}

PreservedAnalyses CombineMMAPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  MMACombiner Combiner;
  if (Combiner.combine(F)) {
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  }

  return PreservedAnalyses::all();
}
