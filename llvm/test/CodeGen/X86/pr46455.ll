; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=avx512vl,avx512bw,avx512dq | FileCheck %s

define void @EntryModule(i8** %buffer_table) {
; CHECK-LABEL: EntryModule:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    movq (%rdi), %rax
; CHECK-NEXT:    movq 24(%rdi), %rcx
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    vcmpneqps (%rax), %xmm0, %xmm0
; CHECK-NEXT:    vpsrld $31, %xmm0, %xmm1
; CHECK-NEXT:    vpshufd {{.*#+}} xmm2 = xmm1[1,1,1,1]
; CHECK-NEXT:    vpshufd {{.*#+}} xmm3 = xmm1[2,3,2,3]
; CHECK-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[3,3,3,3]
; CHECK-NEXT:    vpaddd %xmm1, %xmm3, %xmm1
; CHECK-NEXT:    vpsubd %xmm0, %xmm2, %xmm0
; CHECK-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vmovd %xmm0, (%rcx)
; CHECK-NEXT:    retq
entry:
  %i = bitcast i8** %buffer_table to <8 x float>**
  %i1 = load <8 x float>*, <8 x float>** %i, align 8
  %i6 = load <8 x float>, <8 x float>* %i1, align 16
  %i7 = fcmp une <8 x float> %i6, zeroinitializer
  %i8 = zext <8 x i1> %i7 to <8 x i32>
  %i18 = getelementptr inbounds i8*, i8** %buffer_table, i64 3
  %i19 = load i8*, i8** %i18, align 8
  %shift = shufflevector <8 x i32> %i8, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %i20 = add nuw nsw <8 x i32> %shift, %i8
  %shift13 = shufflevector <8 x i32> %i8, <8 x i32> undef, <8 x i32> <i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %i21 = add nuw nsw <8 x i32> %i20, %shift13
  %shift14 = shufflevector <8 x i32> %i8, <8 x i32> undef, <8 x i32> <i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %i22 = add nuw nsw <8 x i32> %i21, %shift14
  %i23 = extractelement <8 x i32> %i22, i32 0
  %i24 = bitcast i8* %i19 to i32*
  store i32 %i23, i32* %i24, align 8
  ret void
}
