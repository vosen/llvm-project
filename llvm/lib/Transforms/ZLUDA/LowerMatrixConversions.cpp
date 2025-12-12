#include "llvm/Transforms/ZLUDA/LowerMatrixConversions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include <utility>

using namespace llvm;

std::pair<Value *, uint32_t>
getLogicalCoordinatesForAMatrixAMDPhysicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Lane,
                                                      uint32_t vGPR) {
  // A i: (lane % 16)
  Value *Row = Builder.CreateAnd(Lane, 15, "row");

  // A k: 2 * GPR_num + floor(GPR_bits / 16)
  uint32_t Column = 2 * vGPR;

  return {Row, Column};
}

std::pair<uint32_t, Value *>
getLogicalCoordinatesForBMatrixAMDPhysicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Lane,
                                                      uint32_t vGPR) {
  auto [Row, Column] = getLogicalCoordinatesForAMatrixAMDPhysicalCoordinates(
      Builder, Lane, vGPR);

  return {Column, Row};
}

std::pair<Value *, Value *>
getLogicalCoordinatesForCMatrixAMDPhysicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Lane,
                                                      uint32_t vGPR) {
  // C or D i: 2 * GPR_num + floor(lane / 16)
  Value *Row = Builder.CreateAdd(Builder.getInt32(2 * vGPR),
                                 Builder.CreateLShr(Lane, 4), "row");

  // C or D j: (lane % 16)
  Value *Column = Builder.CreateAnd(Lane, 15, "column");

  return {Row, Column};
}

std::pair<Value *, Value *>
getLogicalCoordinatesForDMatrixNVPhysicalCoordinates(IRBuilder<> &Builder,
                                                     Value *Lane,
                                                     uint32_t ElementIdx) {
  // groupID = %laneid >> 2
  Value *GroupID = Builder.CreateLShr(Lane, 2, "group.id");
  // threadID_in_group = %laneid % 4
  Value *ThreadIDInGroup = Builder.CreateAnd(Lane, 3, "thread.id.in.group");

  // row = groupID + (i / 2) * 8
  Value *Row =
      Builder.CreateAdd(GroupID, Builder.getInt32((ElementIdx / 2) * 8), "row");

  // col = (threadID_in_group * 2) + (i & 0x1)
  Value *Column = Builder.CreateAdd(Builder.CreateShl(ThreadIDInGroup, 1),
                                    Builder.getInt32(ElementIdx % 2), "column");

  return {Row, Column};
}

// TODO: refactor shared logic with
// getBMatrixPhysicalCoordinatesForLogicalCoordinates
std::pair<Value *, Value *>
getAMatrixNVPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                     Value *Row,
                                                     uint32_t Column) {
  // groupID = row % 8
  Value *GroupID = Builder.CreateAnd(Row, 7);

  uint32_t ThreadIDInGroup = (Column % 8) / 2;

  // laneid = groupID * 4 + threadID_in_group
  Value *Lane = Builder.CreateAdd(Builder.CreateShl(GroupID, 2),
                                  Builder.getInt32(ThreadIDInGroup), "nv.lane");

  // PackedIdx = (row / 8) + (col / 8) * 2
  Value *PackedIdx =
      Builder.CreateAdd(Builder.CreateLShr(Row, 3),
                        Builder.getInt32((Column >> 3) * 2), "nv.idx");

  return {Lane, PackedIdx};
}

std::tuple<Value *, uint32_t, Value *>
getBMatrixNVPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                     uint32_t Row,
                                                     Value *Column) {
  Value *IsInFirst =
      Builder.CreateICmpULT(Column, Builder.getInt32(8), "is.in.first");

  // groupID = col % 8
  Value *GroupID = Builder.CreateAnd(Column, 7);

  // threadID_in_group = (row % 8) / 2
  uint32_t ThreadIDInGroup = (Row % 8) / 2;

  // laneid = groupID * 4 + ThreadIDInGroup
  Value *Lane = Builder.CreateAdd(Builder.CreateShl(GroupID, 2),
                                  Builder.getInt32(ThreadIDInGroup), "nv.lane");

  // PackedIdx = row / 8
  uint32_t ElementIdx = Row / 8;

  return {Lane, ElementIdx, IsInFirst};
}

std::tuple<Value *, Value *, Value *>
getCMatrixNVPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                     Value *Row,
                                                     Value *Column) {
  Value *IsInFirst =
      Builder.CreateICmpULT(Column, Builder.getInt32(8), "is.in.first");

  // groupID = row % 8
  Value *GroupID = Builder.CreateAnd(Row, 7);

  // threadID_in_group = (col % 8) / 2
  Value *ThreadIDInGroup =
      Builder.CreateLShr(Builder.CreateAnd(Column, 7), 1, "thread.id.in.group");

  // laneid = groupID * 4 + ThreadIDInGroup
  Value *Lane = Builder.CreateAdd(Builder.CreateShl(GroupID, 2),
                                  ThreadIDInGroup, "nv.lane");

  // i = (row / 8) * 2 + col % 2
  Value *ElementIdx =
      Builder.CreateAdd(Builder.CreateShl(Builder.CreateLShr(Row, 3), 1),
                        Builder.CreateAnd(Column, 1));

  return {Lane, ElementIdx, IsInFirst};
}

std::pair<Value *, Value *>
getDMatrixAMDPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Row,
                                                      Value *Column) {
  // C or D[i][j] GPR: floor(i / 2)
  Value *vGPR = Builder.CreateLShr(Row, 1, "element.idx");

  // C or D[i][j] Lane: ((16 * i) % 32) + j
  Value *Lane = Builder.CreateAdd(
      Builder.CreateAnd(Builder.CreateShl(Row, 4), 31), Column, "amd.lane");

  return {Lane, vGPR};
}

static IntrinsicInst *getMatrixConversion(Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (II) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::zluda_amatrix_convert_amd_nv16x16:
    case Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16:
    case Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8:
    case Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8:
    case Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16:
    case Intrinsic::zluda_amatrix_split_amd16x16_nv16x32:
    case Intrinsic::zluda_bmatrix_reshape_amd16x16_nv32x8:
      return II;
    }
  }
  return nullptr;
}

static Value *bpermuteLane(IRBuilder<> &Builder, Value *Lane, Value *X,
                           const Twine &Name = "") {
  if (auto *C = dyn_cast<Constant>(X)) {
    return C;
  }

  Value *BitCast = Builder.CreateBitCast(X, Builder.getInt32Ty());

  Value *Permuted = Builder.CreateIntrinsic(
      Builder.getInt32Ty(), Intrinsic::amdgcn_ds_bpermute,
      {Builder.CreateShl(Lane, 2), BitCast}, nullptr, Name);

  return Builder.CreateBitCast(Permuted, X->getType());
}

class LowerMatrixConversions {
public:
  LowerMatrixConversions(Function &F) : F(F), LaneID(nullptr) {}
  bool lower();

private:
  bool lowerBB(BasicBlock &BB);
  void lowerConversion(IntrinsicInst *Conversion);

  Value *aMatrixConvert(IRBuilder<> &Builder, Value *NVFragment);
  Value *aMatrixConvertI8Half(IRBuilder<> &Builder, Value *NVLowReg,
                              Value *NVHighReg);
  Value *bMatrixConcatenate(IRBuilder<> &Builder, Value *NVFragmentFirst,
                            Value *NVFragmentSecond);
  Value *cMatrixConcatenate(IRBuilder<> &Builder, Value *NVFragmentFirst,
                            Value *NVFragmentSecond);
  Value *dMatrixSplit(IRBuilder<> &Builder, Value *AMDFragment, bool Truncate);

  Value *getLaneNumber();

  Function &F;
  Value *LaneID;
};

Value *LowerMatrixConversions::getLaneNumber() {
  if (LaneID == nullptr) {
    IRBuilder<> Builder(&*F.getEntryBlock().begin());
    LaneID =
        Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {},
                                {Builder.getInt32(-1), Builder.getInt32(0)});
  }

  return LaneID;
}

Value *LowerMatrixConversions::aMatrixConvert(IRBuilder<> &Builder,
                                              Value *NVFragment) {
  Value *Lane = getLaneNumber();

  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false));

  for (uint32_t vGPR = 0; vGPR < 8; ++vGPR) {
    // AMD and NVIDIA formats will have the same packed bf16x2, so only need to
    // find coordinates for the first.
    auto [Row, Column] = getLogicalCoordinatesForAMatrixAMDPhysicalCoordinates(
        Builder, Lane, vGPR);

    auto [CudaTID, NVPackedIdx] =
        getAMatrixNVPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                             Column);

    // Extract the base packed value (even index: 0 or 2)
    Value *BasePackedIdx =
        Builder.CreateAnd(NVPackedIdx, Builder.getInt32(~1), "base.packed.idx");

    // reg0 = NVFragment[basePackedIdx], reg1 = NVFragment[basePackedIdx + 1]
    Value *Reg0 = Builder.CreateExtractElement(NVFragment, BasePackedIdx);
    Value *Reg1 = Builder.CreateExtractElement(
        NVFragment, Builder.CreateAdd(BasePackedIdx, Builder.getInt32(1)));

    // a_tmp0 = bpermute_lane(cudaTID, reg0)
    Value *ATmp0 = bpermuteLane(Builder, CudaTID, Reg0, "a.tmp0");

    // a_tmp1 = bpermute_lane(cudaTID, reg1)
    Value *ATmp1 = bpermuteLane(Builder, CudaTID, Reg1, "a.tmp1");

    // a_Frag_reg = (lane < 8) ? a_tmp0 : a_tmp1
    Value *LaneMod16 = Builder.CreateAnd(Lane, 15, "lane.mod16");
    Value *LaneLT8 =
        Builder.CreateICmpULT(LaneMod16, Builder.getInt32(8), "lane.lt.8");
    Value *AFragReg = Builder.CreateSelect(LaneLT8, ATmp0, ATmp1, "a.frag.reg");

    // Extract bottom 16 bits: aFrag[2 * vGPR] = bottom16AsFp16(a_Frag_reg)
    Value *Bottom16 =
        Builder.CreateTrunc(AFragReg, Builder.getInt16Ty(), "bottom16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Bottom16, 2 * vGPR);

    // Extract top 16 bits: aFrag[2 * vGPR + 1] = top16AsFp16(a_Frag_reg)
    Value *Top16 = Builder.CreateTrunc(Builder.CreateLShr(AFragReg, 16),
                                       Builder.getInt16Ty(), "top16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Top16, 2 * vGPR + 1);
  }

  return AMDFragment;
}

Value *LowerMatrixConversions::aMatrixConvertI8Half(IRBuilder<> &Builder,
                                                    Value *NvLowReg,
                                                    Value *NvHighReg) {
  Value *Lane = getLaneNumber();
  Value *QuarterLane = Builder.CreateAnd(Lane, 7, "quarter.lane");
  Value *HalfLane = Builder.CreateAnd(Lane, 15, "half.lane");
  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false));

  Value *UsesLowSrc = Builder.CreateCmp(llvm::CmpInst::ICMP_ULT, HalfLane,
                                        Builder.getInt32(8), "uses.low.src");
  for (uint32_t vGPR = 0; vGPR < 4; ++vGPR) {
    Value *SrcThread =
        Builder.CreateAdd(Builder.CreateMul(QuarterLane, Builder.getInt32(4)),
                          Builder.getInt32(vGPR), "src.thread");
    Value *APermutedLow = bpermuteLane(Builder, SrcThread, NvLowReg);
    Value *APermutedHigh = bpermuteLane(Builder, SrcThread, NvHighReg);
    Value *APermuted =
        Builder.CreateSelect(UsesLowSrc, APermutedLow, APermutedHigh);
    AMDFragment = Builder.CreateInsertElement(AMDFragment, APermuted, vGPR);
  }
  return AMDFragment;
}

Value *LowerMatrixConversions::bMatrixConcatenate(IRBuilder<> &Builder,
                                                  Value *NVFragmentFirst,
                                                  Value *NVFragmentSecond) {
  Value *Lane = getLaneNumber();

  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false));

  for (uint32_t vGPR = 0; vGPR < 8; ++vGPR) {
    // AMD and NVIDIA formats will have the same packed bf16x2, so only need to
    // find coordinates for the first.
    auto [Row, Column] = getLogicalCoordinatesForBMatrixAMDPhysicalCoordinates(
        Builder, Lane, vGPR);

    auto [CudaTID, NVPackedIdx, IsInFirst] =
        getBMatrixNVPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                             Column);

    Value *RegFirst =
        Builder.CreateExtractElement(NVFragmentFirst, NVPackedIdx);
    Value *RegSecond =
        Builder.CreateExtractElement(NVFragmentSecond, NVPackedIdx);

    // b_Frag_reg = bpermute_lane(cudaTID, reg)
    Value *BFragRegFirst =
        bpermuteLane(Builder, CudaTID, RegFirst, "b.frag.reg.first");
    Value *BFragRegSecond =
        bpermuteLane(Builder, CudaTID, RegSecond, "b.frag.reg.second");

    // Extract bottom 16 bits
    Value *Bottom16First = Builder.CreateTrunc(
        BFragRegFirst, Builder.getInt16Ty(), "bottom16.first");
    Value *Bottom16Second = Builder.CreateTrunc(
        BFragRegSecond, Builder.getInt16Ty(), "bottom16.second");
    Value *Bottom16 = Builder.CreateSelect(IsInFirst, Bottom16First,
                                           Bottom16Second, "bottom16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Bottom16, 2 * vGPR);

    // Extract top 16 bits
    Value *Top16First =
        Builder.CreateTrunc(Builder.CreateLShr(BFragRegFirst, 16),
                            Builder.getInt16Ty(), "top16.first");
    Value *Top16Second =
        Builder.CreateTrunc(Builder.CreateLShr(BFragRegSecond, 16),
                            Builder.getInt16Ty(), "top16.second");
    Value *Top16 =
        Builder.CreateSelect(IsInFirst, Top16First, Top16Second, "top16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Top16, 2 * vGPR + 1);
  }

  return AMDFragment;
}

Value *LowerMatrixConversions::cMatrixConcatenate(IRBuilder<> &Builder,
                                                  Value *NVFragmentFirst,
                                                  Value *NVFragmentSecond) {
  auto *RetTy = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);

  if (Constant *C0 = dyn_cast<Constant>(NVFragmentFirst)) {
    if (Constant *C1 = dyn_cast<Constant>(NVFragmentSecond)) {
      if (C0->isZeroValue() && C1->isZeroValue()) {
        return Constant::getNullValue(RetTy);
      }
    }
  }

  Value *Lane = getLaneNumber();

  Value *AMDFragment = PoisonValue::get(RetTy);

  for (uint32_t vGPR = 0; vGPR < 8; ++vGPR) {
    auto [Row, Column] = getLogicalCoordinatesForCMatrixAMDPhysicalCoordinates(
        Builder, Lane, vGPR);

    auto [CudaTID, NVElementIdx, IsInFirst] =
        getCMatrixNVPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                             Column);

    Value *BaseIdx =
        Builder.CreateAnd(NVElementIdx, Builder.getInt32(~1), "base.idx");

    Value *Ctmp0SrcFirst =
        Builder.CreateExtractElement(NVFragmentFirst, BaseIdx);
    Value *Ctmp0SrcSecond =
        Builder.CreateExtractElement(NVFragmentSecond, BaseIdx);
    Value *Ctmp1SrcFirst = Builder.CreateExtractElement(
        NVFragmentFirst, Builder.CreateAdd(BaseIdx, Builder.getInt32(1)));
    Value *Ctmp1SrcSecond = Builder.CreateExtractElement(
        NVFragmentSecond, Builder.CreateAdd(BaseIdx, Builder.getInt32(1)));

    // ctmp0 = bpermute_lane(cudaTID, ...)
    Value *Ctmp0First =
        bpermuteLane(Builder, CudaTID, Ctmp0SrcFirst, "ctmp0.first");
    Value *Ctmp0Second =
        bpermuteLane(Builder, CudaTID, Ctmp0SrcSecond, "ctmp0.second");

    // ctmp1 = bpermute_lane(cudaTID, ...)
    Value *Ctmp1First =
        bpermuteLane(Builder, CudaTID, Ctmp1SrcFirst, "ctmp1.first");
    Value *Ctmp1Second =
        bpermuteLane(Builder, CudaTID, Ctmp1SrcSecond, "ctmp1.second");

    // cFrag[vGPR] = (lIdx & 1) ? ctmp1 : ctmp0
    Value *LaneIsOdd = Builder.CreateTrunc(Builder.CreateAnd(Lane, 1),
                                           Builder.getInt1Ty(), "lane.is.odd");
    Value *CFragValueFirst = Builder.CreateSelect(
        LaneIsOdd, Ctmp1First, Ctmp0First, "cfrag.value.first");
    Value *CFragValueSecond = Builder.CreateSelect(
        LaneIsOdd, Ctmp1Second, Ctmp0Second, "cfrag.value.second");

    Value *FinalValue = Builder.CreateSelect(IsInFirst, CFragValueFirst,
                                             CFragValueSecond, "final.value");

    AMDFragment = Builder.CreateInsertElement(AMDFragment, FinalValue, vGPR);
  }

  return AMDFragment;
}

Value *LowerMatrixConversions::dMatrixSplit(IRBuilder<> &Builder,
                                            Value *AMDFragment, bool Truncate) {
  Value *Lane = getLaneNumber();

  Value *NVFragmentFirst = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false));

  for (uint32_t cChunk = 0; cChunk < 4; ++cChunk) {
    auto [Row, ColumnFirst] =
        getLogicalCoordinatesForDMatrixNVPhysicalCoordinates(Builder, Lane,
                                                             cChunk);

    auto [R_lIdxFirst, AMDvGPRFirst] =
        getDMatrixAMDPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                              ColumnFirst);

    //  r_vGPR = (AMDvGPR / 4) * 4
    Value *BaseVGPRFirst = Builder.CreateAnd(AMDvGPRFirst, Builder.getInt32(~3),
                                             "base.vgpr.first");

    // d_tmp0 = bpermute_lane(r_lIdx, dFrag[baseVGPR])
    Value *DTmp0First =
        bpermuteLane(Builder, R_lIdxFirst,
                     Builder.CreateExtractElement(AMDFragment, BaseVGPRFirst),
                     "d.tmp0.first");

    // d_tmp1 = bpermute_lane(r_lIdx, dFrag[baseVGPR + 1])
    Value *DTmp1First = bpermuteLane(
        Builder, R_lIdxFirst,
        Builder.CreateExtractElement(
            AMDFragment, Builder.CreateAdd(BaseVGPRFirst, Builder.getInt32(1))),
        "d.tmp1.first");

    // d_tmp2 = bpermute_lane(r_lIdx, dFrag[baseVGPR + 2])
    Value *DTmp2First = bpermuteLane(
        Builder, R_lIdxFirst,
        Builder.CreateExtractElement(
            AMDFragment, Builder.CreateAdd(BaseVGPRFirst, Builder.getInt32(2))),
        "d.tmp2.first");

    // d_tmp3 = bpermute_lane(r_lIdx, dFrag[baseVGPR + 3])
    Value *DTmp3First = bpermuteLane(
        Builder, R_lIdxFirst,
        Builder.CreateExtractElement(
            AMDFragment, Builder.CreateAdd(BaseVGPRFirst, Builder.getInt32(3))),
        "d.tmp3.first");

    // if (lIdx < 8) val = d_tmp0;
    // else if (lIdx < 16) val = d_tmp1;
    // else if (lIdx < 24) val = d_tmp2;
    // else val = d_tmp3;
    Value *LaneLT8 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(8), "lane.lt.8");
    Value *LaneLT16 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(16), "lane.lt.16");
    Value *LaneLT24 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(24), "lane.lt.24");

    // Build nested selects: lane < 8 ? tmp0 : (lane < 16 ? tmp1 : (lane < 24
    // ? tmp2 : tmp3))
    Value *Select23First = Builder.CreateSelect(LaneLT24, DTmp2First,
                                                DTmp3First, "select.23.first");
    Value *Select123First = Builder.CreateSelect(
        LaneLT16, DTmp1First, Select23First, "select.123.first");
    Value *ValFirst =
        Builder.CreateSelect(LaneLT8, DTmp0First, Select123First, "val.first");

    NVFragmentFirst =
        Builder.CreateInsertElement(NVFragmentFirst, ValFirst, cChunk);
  }

  if (Truncate) {
    return NVFragmentFirst;
  }

  Value *NVFragmentSecond = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false));

  for (uint32_t cChunk = 0; cChunk < 4; ++cChunk) {
    auto [Row, ColumnFirst] =
        getLogicalCoordinatesForDMatrixNVPhysicalCoordinates(Builder, Lane,
                                                             cChunk);

    // Process the second fragment (column + 8)
    Value *ColumnSecond =
        Builder.CreateAdd(ColumnFirst, Builder.getInt32(8), "column.second");
    auto [R_lIdxSecond, AMDvGPRSecond] =
        getDMatrixAMDPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                              ColumnSecond);

    Value *BaseVGPRSecond = Builder.CreateAnd(
        AMDvGPRSecond, Builder.getInt32(~3), "base.vgpr.second");

    Value *DTmp0Second =
        bpermuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(AMDFragment, BaseVGPRSecond),
                     "d.tmp0.second");

    Value *DTmp1Second =
        bpermuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(
                         AMDFragment, Builder.CreateAdd(BaseVGPRSecond,
                                                        Builder.getInt32(1))),
                     "d.tmp1.second");

    Value *DTmp2Second =
        bpermuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(
                         AMDFragment, Builder.CreateAdd(BaseVGPRSecond,
                                                        Builder.getInt32(2))),
                     "d.tmp2.second");

    Value *DTmp3Second =
        bpermuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(
                         AMDFragment, Builder.CreateAdd(BaseVGPRSecond,
                                                        Builder.getInt32(3))),
                     "d.tmp3.second");

    Value *LaneLT8 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(8), "lane.lt.8");
    Value *LaneLT16 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(16), "lane.lt.16");
    Value *LaneLT24 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(24), "lane.lt.24");

    Value *Select23Second = Builder.CreateSelect(
        LaneLT24, DTmp2Second, DTmp3Second, "select.23.second");
    Value *Select123Second = Builder.CreateSelect(
        LaneLT16, DTmp1Second, Select23Second, "select.123.second");
    Value *ValSecond = Builder.CreateSelect(LaneLT8, DTmp0Second,
                                            Select123Second, "val.second");

    NVFragmentSecond =
        Builder.CreateInsertElement(NVFragmentSecond, ValSecond, cChunk);
  }

  Type *StructTy =
      StructType::get(NVFragmentFirst->getType(), NVFragmentSecond->getType());
  Value *Result = PoisonValue::get(StructTy);
  Result = Builder.CreateInsertValue(Result, NVFragmentFirst, 0);
  Result = Builder.CreateInsertValue(Result, NVFragmentSecond, 1);

  return Result;
}

void LowerMatrixConversions::lowerConversion(IntrinsicInst *Conversion) {
  IRBuilder<> Builder(Conversion);

  switch (Conversion->getIntrinsicID()) {
  case Intrinsic::zluda_amatrix_convert_amd_nv16x16:
    Conversion->replaceAllUsesWith(
        aMatrixConvert(Builder, Conversion->getArgOperand(0)));
    Conversion->eraseFromParent();
    break;
  case Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8: {
    Value *NVMatrixFirst = Conversion->getArgOperand(0);
    Value *NVMatrixSecond = Conversion->getArgOperand(1);
    Conversion->replaceAllUsesWith(
        bMatrixConcatenate(Builder, NVMatrixFirst, NVMatrixSecond));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8: {
    Value *NVMatrixFirst = Conversion->getArgOperand(0);
    Value *NVMatrixSecond = Conversion->getArgOperand(1);
    Conversion->replaceAllUsesWith(
        cMatrixConcatenate(Builder, NVMatrixFirst, NVMatrixSecond));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16: {
    Value *AMDMatrix = Conversion->getArgOperand(0);
    auto AMDMatrixBitCast = Builder.CreateBitCast(
        AMDMatrix,
        VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false));
    Conversion->replaceAllUsesWith(
        dMatrixSplit(Builder, AMDMatrixBitCast, /*Truncate=*/true));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16: {
    Value *AMDMatrix = Conversion->getArgOperand(0);
    Conversion->replaceAllUsesWith(
        dMatrixSplit(Builder, AMDMatrix, /*Truncate=*/false));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_amatrix_split_amd16x16_nv16x32: {
    Value *NVMatrix = Conversion->getArgOperand(0);
    auto *V0 = Builder.CreateExtractElement(NVMatrix, uint64_t(0));
    auto *V1 = Builder.CreateExtractElement(NVMatrix, uint64_t(1));
    auto *V2 = Builder.CreateExtractElement(NVMatrix, uint64_t(2));
    auto *V3 = Builder.CreateExtractElement(NVMatrix, uint64_t(3));
    auto LowHalf = aMatrixConvertI8Half(Builder, V0, V1);
    auto HighHalf = aMatrixConvertI8Half(Builder, V2, V3);
    Type *ReturnTy = StructType::get(LowHalf->getType(), HighHalf->getType());
    Value *Result = PoisonValue::get(ReturnTy);
    Result = Builder.CreateInsertValue(Result, LowHalf, 0);
    Result = Builder.CreateInsertValue(Result, HighHalf, 1);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_bmatrix_reshape_amd16x16_nv32x8: {
    Value *LeftNVMatrix = Conversion->getArgOperand(0);
    Value *RightNVMatrix = Conversion->getArgOperand(1);
    Value *UpperLeftMatrix =
        Builder.CreateExtractElement(LeftNVMatrix, uint64_t(0));
    Value *UpperRightMatrix =
        Builder.CreateExtractElement(RightNVMatrix, uint64_t(0));
    Value *LowerLeftMatrix =
        Builder.CreateExtractElement(LeftNVMatrix, uint64_t(1));
    Value *LowerRightMatrix =
        Builder.CreateExtractElement(RightNVMatrix, uint64_t(1));
    Value *B0 =
        aMatrixConvertI8Half(Builder, UpperLeftMatrix, UpperRightMatrix);
    Value *B1 =
        aMatrixConvertI8Half(Builder, LowerLeftMatrix, LowerRightMatrix);
    Type *ReturnTy = StructType::get(B0->getType(), B1->getType());
    Value *Result = PoisonValue::get(ReturnTy);
    Result = Builder.CreateInsertValue(Result, B0, 0);
    Result = Builder.CreateInsertValue(Result, B1, 1);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  default:
    llvm_unreachable("Unexpected intrinsic");
  }
}

bool LowerMatrixConversions::lowerBB(BasicBlock &BB) {
  bool Modified = false;

  for (Instruction &I : make_early_inc_range(BB)) {
    auto Conversion = getMatrixConversion(I);

    if (Conversion) {
      lowerConversion(Conversion);
      Modified = true;
    }
  }

  return Modified;
}

bool LowerMatrixConversions::lower() {
  bool Modified = false;

  for (BasicBlock &BB : F) {
    Modified |= lowerBB(BB);
  }

  return Modified;
}

PreservedAnalyses LowerMatrixConversionsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  LowerMatrixConversions LMC(F);
  if (LMC.lower()) {
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  }

  return PreservedAnalyses::all();
}
