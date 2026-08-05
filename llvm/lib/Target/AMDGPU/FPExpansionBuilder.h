#ifndef LLVM_LIB_TARGET_AMDGPU_FPEXPANSIONBUILDER_H
#define LLVM_LIB_TARGET_AMDGPU_FPEXPANSIONBUILDER_H

#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {

class SelectionDAG;

// ZLUDA changes start
class FPExpansionBuilder {
  SelectionDAG &DAG;
  SDValue Op;
  SDValue Chain;

public:
  FPExpansionBuilder(SDValue Op, SelectionDAG &DAG);

  /// \returns true when a strict node is being expanded, i.e. when the emitted
  /// nodes are chained.
  bool isStrict() const;

  /// \returns operand \p I of the node being expanded, skipping the chain of a
  /// strict node.
  SDValue getOperand(unsigned I) const;

  /// Pair \p Result up with the final chain when expanding a strict node, so
  /// that it can be returned from LowerOperation directly.
  SDValue finish(const SDLoc &SL, SDValue Result);

  SDValue fmul(const SDLoc &SL, EVT VT, SDValue A, SDValue B);
  SDValue fma(const SDLoc &SL, EVT VT, SDValue A, SDValue B, SDValue C);
  SDValue fldexp(const SDLoc &SL, EVT VT, SDValue A, SDValue Exp);
  SDValue rcpNoflags(const SDLoc &SL, EVT VT, SDValue A);
  SDValue rsq(const SDLoc &SL, EVT VT, SDValue A);
  SDValue sqrt(const SDLoc &SL, EVT VT, SDValue A);

  /// The INT_TO_FP expansions do not propagate the flags of the node being
  /// expanded, matching the non-strict code they were derived from, so these
  /// emit with default flags rather than with Op's.
  SDValue intToFPNoflags(const SDLoc &SL, EVT VT, SDValue A, bool Signed);
  SDValue fldexpNoflags(const SDLoc &SL, EVT VT, SDValue A, SDValue Exp);
  SDValue faddNoflags(const SDLoc &SL, EVT VT, SDValue A, SDValue B);
  SDValue fpRoundNoflags(const SDLoc &SL, EVT VT, SDValue A, SDValue Flag);

private:
  SDValue emit(unsigned Opc, unsigned StrictOpc, const SDLoc &SL, EVT VT,
               ArrayRef<SDValue> Ops);
  SDValue emit(unsigned Opc, unsigned StrictOpc, const SDLoc &SL, EVT VT,
               ArrayRef<SDValue> Ops, SDNodeFlags Flags);
  SDValue emitNoflags(unsigned Opc, unsigned StrictOpc, const SDLoc &SL, EVT VT,
                      ArrayRef<SDValue> Ops);
};
// ZLUDA changes end

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_FPEXPANSIONBUILDER_H
