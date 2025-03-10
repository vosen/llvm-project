; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Check that the booleans are converted using zext and not via sext.
; 0x1 means that we only look at the first bit.

define void @ui_to_fp_conv(<8 x float> * nocapture %aFOO, <8 x float>* nocapture %RET) nounwind {
; CHECK-LABEL: ui_to_fp_conv:
; CHECK:       # %bb.0: # %allocas
; CHECK-NEXT:    movsd {{.*#+}} xmm0 = [1.0E+0,1.0E+0,0.0E+0,0.0E+0]
; CHECK-NEXT:    xorps %xmm1, %xmm1
; CHECK-NEXT:    movups %xmm1, 16(%rsi)
; CHECK-NEXT:    movups %xmm0, (%rsi)
; CHECK-NEXT:    retq
allocas:
  %bincmp = fcmp olt <8 x float> <float 1.000000e+00, float 1.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00> , <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %bool2float = uitofp <8 x i1> %bincmp to <8 x float>
  store <8 x float> %bool2float, <8 x float>* %RET, align 4
  ret void
}



