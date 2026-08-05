#include "FPExpansionBuilder.h"
#include "AMDGPUISelLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

// ZLUDA changes start
FPExpansionBuilder::FPExpansionBuilder(SDValue Op, SelectionDAG &DAG)
    : DAG(DAG), Op(Op),
      Chain(Op->isStrictFPOpcode() ? Op.getOperand(0) : SDValue()) {}

bool FPExpansionBuilder::isStrict() const { return Chain.getNode() != nullptr; }

SDValue FPExpansionBuilder::getOperand(unsigned I) const {
  return Op.getOperand(Op->isStrictFPOpcode() ? I + 1 : I);
}

SDValue FPExpansionBuilder::finish(const SDLoc &SL, SDValue Result) {
  if (!Chain)
    return Result;
  return DAG.getMergeValues({Result, Chain}, SL);
}

SDValue FPExpansionBuilder::fmul(const SDLoc &SL, EVT VT, SDValue A,
                                 SDValue B) {
  return emit(ISD::FMUL, ISD::STRICT_FMUL, SL, VT, {A, B});
}

SDValue FPExpansionBuilder::fma(const SDLoc &SL, EVT VT, SDValue A, SDValue B,
                                SDValue C) {
  return emit(ISD::FMA, ISD::STRICT_FMA, SL, VT, {A, B, C});
}

SDValue FPExpansionBuilder::fldexp(const SDLoc &SL, EVT VT, SDValue A,
                                   SDValue Exp) {
  return emit(ISD::FLDEXP, ISD::STRICT_FLDEXP, SL, VT, {A, Exp});
}

SDValue FPExpansionBuilder::rcpNoflags(const SDLoc &SL, EVT VT, SDValue A) {
  return emit(AMDGPUISD::RCP, AMDGPUISD::STRICT_RCP, SL, VT, {A},
              SDNodeFlags());
}

SDValue FPExpansionBuilder::rsq(const SDLoc &SL, EVT VT, SDValue A) {
  return emit(AMDGPUISD::RSQ, AMDGPUISD::STRICT_RSQ, SL, VT, {A});
}

SDValue FPExpansionBuilder::sqrt(const SDLoc &SL, EVT VT, SDValue A) {
  if (!Chain) {
    SDValue SqrtID =
        DAG.getTargetConstant(Intrinsic::amdgcn_sqrt, SL, MVT::i32);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SL, VT, SqrtID, A,
                       Op->getFlags());
  }

  SDValue N =
      DAG.getNode(AMDGPUISD::STRICT_SQRT, SL, DAG.getVTList(VT, MVT::Other),
                  {Chain, A}, Op->getFlags());
  Chain = N.getValue(1);
  return N;
}

SDValue FPExpansionBuilder::intToFPNoflags(const SDLoc &SL, EVT VT, SDValue A,
                                           bool Signed) {
  if (Signed)
    return emitNoflags(ISD::SINT_TO_FP, ISD::STRICT_SINT_TO_FP, SL, VT, {A});
  return emitNoflags(ISD::UINT_TO_FP, ISD::STRICT_UINT_TO_FP, SL, VT, {A});
}

SDValue FPExpansionBuilder::fldexpNoflags(const SDLoc &SL, EVT VT, SDValue A,
                                          SDValue Exp) {
  return emitNoflags(ISD::FLDEXP, ISD::STRICT_FLDEXP, SL, VT, {A, Exp});
}

SDValue FPExpansionBuilder::faddNoflags(const SDLoc &SL, EVT VT, SDValue A,
                                        SDValue B) {
  return emitNoflags(ISD::FADD, ISD::STRICT_FADD, SL, VT, {A, B});
}

SDValue FPExpansionBuilder::fpRoundNoflags(const SDLoc &SL, EVT VT, SDValue A,
                                           SDValue Flag) {
  return emitNoflags(ISD::FP_ROUND, ISD::STRICT_FP_ROUND, SL, VT, {A, Flag});
}

SDValue FPExpansionBuilder::emit(unsigned Opc, unsigned StrictOpc,
                                 const SDLoc &SL, EVT VT,
                                 ArrayRef<SDValue> Ops) {
  return emit(Opc, StrictOpc, SL, VT, Ops, Op->getFlags());
}

SDValue FPExpansionBuilder::emit(unsigned Opc, unsigned StrictOpc,
                                 const SDLoc &SL, EVT VT, ArrayRef<SDValue> Ops,
                                 SDNodeFlags Flags) {
  if (!Chain)
    return DAG.getNode(Opc, SL, VT, Ops, Flags);

  SmallVector<SDValue> StrictOps;
  StrictOps.push_back(Chain);
  StrictOps.append(Ops.begin(), Ops.end());

  SDValue N = DAG.getNode(StrictOpc, SL, DAG.getVTList(VT, MVT::Other),
                          StrictOps, Flags);
  Chain = N.getValue(1);
  return N;
}

SDValue FPExpansionBuilder::emitNoflags(unsigned Opc, unsigned StrictOpc,
                                        const SDLoc &SL, EVT VT,
                                        ArrayRef<SDValue> Ops) {
  if (!Chain)
    return DAG.getNode(Opc, SL, VT, Ops);

  SmallVector<SDValue> StrictOps;
  StrictOps.push_back(Chain);
  StrictOps.append(Ops.begin(), Ops.end());

  SDValue N = DAG.getNode(StrictOpc, SL, DAG.getVTList(VT, MVT::Other),
                          StrictOps);
  Chain = N.getValue(1);
  return N;
}
// ZLUDA changes end
