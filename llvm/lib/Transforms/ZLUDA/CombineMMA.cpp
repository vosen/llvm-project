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
        // This is conservative
        if (Dependency->mayReadOrWriteMemory()) {
          return false;
        }
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

static IntrinsicInst *getBF16ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA && MMA->getIntrinsicID() ==
                 Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    return MMA;
  }
  return nullptr;
}

static IntrinsicInst *getS8ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA &&
      MMA->getIntrinsicID() == Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    return MMA;
  }
  return nullptr;
}

class MMACombiner {
public:
  bool combine(Function &F);

private:
  bool combineBB(BasicBlock &BB);
  bool combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs);
  bool combineMMA(IntrinsicInst *First, IntrinsicInst *Second);

  llvm::Value *EmitAmdMmaI8(llvm::IRBuilder<> &Builder, llvm::Value *FirstA,
                            llvm::Value *FirstB, llvm::Value *SecondB,
                            llvm::Value *FirstC, llvm::Value *SecondC);

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
    auto FirstIndices = FirstExtract->getIndices();
    auto *SecondAggregate = SecondExtract->getAggregateOperand();
    auto SecondIndices = SecondExtract->getIndices();
    if (FirstAggregate == SecondAggregate &&
        FirstIndices == ArrayRef<unsigned>{0} &&
        SecondIndices == ArrayRef<unsigned>{1}) {
      if (auto *II = dyn_cast<IntrinsicInst>(FirstAggregate)) {
        if (II->getIntrinsicID() ==
            Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16) {
          MaybeRemove.emplace_back(FirstExtract);
          MaybeRemove.emplace_back(SecondExtract);
          MaybeRemove.emplace_back(II);
          return II->getArgOperand(0);
        }
      }
    }
  }

  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);

  return Builder.CreateIntrinsic(
      V8I32Ty, Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8,
      {FirstC, SecondC});
}

// If C is the result of a truncate, return the value before it was truncated.
// Otherwise zero-extend the matrix.
Value *MMACombiner::convertC(IRBuilder<> &Builder, Value *C) {
  if (auto *II = dyn_cast<IntrinsicInst>(C)) {
    if (II->getIntrinsicID() ==
        Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16) {
      MaybeRemove.emplace_back(II);
      return II->getArgOperand(0);
    }
  }

  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);
  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);

  auto NullC = Constant::getNullValue(C->getType());
  auto IntResult = Builder.CreateIntrinsic(
      V8I32Ty, Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8, {C, NullC});
  return Builder.CreateBitCast(IntResult, V8F32Ty);
}

// Combine two NVIDIA-style 16x8 MMA instructions into one AMD-style 16x16 MMA
// instruction.
bool MMACombiner::combineMMA(IntrinsicInst *First, IntrinsicInst *Second) {
  assert(First->getIntrinsicID() == Second->getIntrinsicID());
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

  llvm::Value *Split;
  if (First->getIntrinsicID() ==
      Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
    auto V4I32x2Ty = StructType::get(Builder.getContext(), {V4I32Ty, V4I32Ty});
    auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);
    auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);
    auto V16I16Ty =
        VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false);

    auto ShuffledA = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_amatrix_convert_amd_nv16x16, {FirstA});
    auto CombinedB = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8,
        {FirstB, SecondB});
    auto *FirstCBitCast = Builder.CreateBitCast(FirstC, V4I32Ty);
    auto *SecondCBitCast = Builder.CreateBitCast(SecondC, V4I32Ty);
    auto CombinedC = combineC(Builder, FirstCBitCast, SecondCBitCast);
    auto CombinedCBitCast = Builder.CreateBitCast(CombinedC, V8F32Ty);

    auto *Result = Builder.CreateIntrinsic(
        V8F32Ty, Intrinsic::amdgcn_wmma_f32_16x16x16_bf16,
        {ShuffledA, CombinedB, CombinedCBitCast});
    auto *ResultBitCast = Builder.CreateBitCast(Result, V8I32Ty);
    Split = Builder.CreateIntrinsic(
        V4I32x2Ty, Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16,
        {ResultBitCast});
  } else if (First->getIntrinsicID() ==
             Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    Split = EmitAmdMmaI8(Builder, FirstA, FirstB, SecondB, FirstC, SecondC);
  } else {
    llvm_unreachable("Unsupported MMA intrinsic");
  }

  auto NewFirst = Builder.CreateExtractValue(Split, {0});
  auto NewSecond = Builder.CreateExtractValue(Split, {1});

  First->replaceAllUsesWith(NewFirst);
  Second->replaceAllUsesWith(NewSecond);

  First->eraseFromParent();
  Second->eraseFromParent();

  return true;
}

llvm::Value *MMACombiner::EmitAmdMmaI8(llvm::IRBuilder<> &Builder,
                                       llvm::Value *A, llvm::Value *FirstB,
                                       llvm::Value *SecondB,
                                       llvm::Value *FirstC,
                                       llvm::Value *SecondC) {
  auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  auto V4I32x2Ty = StructType::get(Builder.getContext(), {V4I32Ty, V4I32Ty});
  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);

  auto SplitA = Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_amatrix_split_amd16x16_nv16x32, {A});
  auto ReshapedB = Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_bmatrix_reshape_amd16x16_nv32x8,
      {FirstB, SecondB});

  auto CombinedC = combineC(Builder, FirstC, SecondC);
  auto A0 = Builder.CreateExtractValue(SplitA, {0});
  auto A1 = Builder.CreateExtractValue(SplitA, {1});
  auto B0 = Builder.CreateExtractValue(ReshapedB, {0});
  auto B1 = Builder.CreateExtractValue(ReshapedB, {1});

  auto True = Builder.getTrue();
  auto False = Builder.getFalse();

  auto TempD =
      Builder.CreateIntrinsic(V8I32Ty, Intrinsic::amdgcn_wmma_i32_16x16x16_iu8,
                              {True, A0, True, B0, CombinedC, False});
  auto D =
      Builder.CreateIntrinsic(V8I32Ty, Intrinsic::amdgcn_wmma_i32_16x16x16_iu8,
                              {True, A1, True, B1, TempD, False});
  return Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16, {D});
}

// Lower an NVIDIA-style 16x8 MMA instruction to an AMD-style 16x16 MMA
// instruction. The unused part of the matrix is filled with zeroes.
void MMACombiner::lowerMMA(IntrinsicInst *MMA) {

  Value *A = MMA->getArgOperand(0);
  Value *B = MMA->getArgOperand(1);
  Value *C = MMA->getArgOperand(2);

  IRBuilder<> Builder(MMA);

  llvm::Value *Result;

  llvm::Intrinsic::ID IID = MMA->getIntrinsicID();
  auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  if (IID == Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);
    auto V16I16Ty =
        VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false);

    auto ShuffledA = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_amatrix_convert_amd_nv16x16, {A});
    auto NullB = Constant::getNullValue(B->getType());
    auto ShuffledB = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8, {B, NullB});
    auto ShuffledC = convertC(Builder, C);

    auto *Output = Builder.CreateIntrinsic(
        V8F32Ty, Intrinsic::amdgcn_wmma_f32_16x16x16_bf16,
        {ShuffledA, ShuffledB, ShuffledC});
    Result = Builder.CreateIntrinsic(
        V4I32Ty, Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16, {Output});
  } else if (IID == Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    auto BPadding = Constant::getNullValue(B->getType());
    auto CPadding = Constant::getNullValue(C->getType());
    llvm::Value *DoubleResult =
        EmitAmdMmaI8(Builder, A, B, BPadding, C, CPadding);
    Result = Builder.CreateExtractValue(DoubleResult, 0);
  } else {
    llvm_unreachable("Unsupported MMA intrinsic");
  }
  MMA->replaceAllUsesWith(Result);
  MMA->eraseFromParent();
}

bool MMACombiner::combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs) {
  bool Modified = !MMAs.empty();

  llvm::DenseMap<std::pair<llvm::Intrinsic::ID, llvm::Value *>, IntrinsicInst *>
      UncombinedMMAs;

  for (IntrinsicInst *MMA : MMAs) {
    std::pair<llvm::Intrinsic::ID, llvm::Value *> Key{MMA->getIntrinsicID(),
                                                      MMA->getArgOperand(0)};
    IntrinsicInst *CompatibleMMA = UncombinedMMAs.lookup(Key);
    if (CompatibleMMA) {
      if (combineMMA(CompatibleMMA, MMA)) {
        UncombinedMMAs.erase(Key);
      } else {
        // If we failed that's likely because the MMA #2 depends on MMA #1.
        // In that case we lower MMA #1 and keep MMA #2 for future combinations.
        lowerMMA(CompatibleMMA);
        UncombinedMMAs.insert_or_assign(Key, MMA);
      }
    } else {
      UncombinedMMAs.insert_or_assign(Key, MMA);
    }
  }

  for (auto pair : UncombinedMMAs) {
    lowerMMA(pair.second);
  }

  return Modified;
}

bool MMACombiner::combineBB(BasicBlock &BB) {
  // For now, we simply combine adjacent m16n8k16 MMAs if possible. This may be
  // good enough in most cases. Any MMAs that cannot be combined are lowered
  // individually.
  bool Modified = false;

  SmallVector<IntrinsicInst *> MMAs;

  for (Instruction &I : BB) {
    auto *BF16MMA = getBF16ZludaMMA(I);
    if (BF16MMA) {
      MMAs.push_back(BF16MMA);
      continue;
    }
    auto *S8MMA = getS8ZludaMMA(I);
    if (S8MMA) {
      MMAs.push_back(S8MMA);
      continue;
    }
  }

  Modified |= combineMMAs(MMAs);

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
