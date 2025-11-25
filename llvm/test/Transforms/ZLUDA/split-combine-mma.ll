; RUN: opt -S -passes=zluda-split-mma,zluda-combine-mma -mtriple=amdgcn-amd-amdhsa %s | FileCheck %s

; CHECK-LABEL: @split_combine
; CHECK-NEXT: [[a_split:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.zluda.amatrix.split.nv16x32(<4 x i32> %a)
; CHECK-NEXT: [[a0:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[a_split]], 0
; CHECK-NEXT: [[a1:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[a_split]], 1
; CHECK-NEXT: [[b_split:%.*]] = call { <1 x i32>, <1 x i32> } @llvm.zluda.bmatrix.split.nv32x8(<2 x i32> %b)
; CHECK-NEXT: [[b0:%.*]] = extractvalue { <1 x i32>, <1 x i32> } [[b_split]], 0
; CHECK-NEXT: [[b1:%.*]] = extractvalue { <1 x i32>, <1 x i32> } [[b_split]], 1
; CHECK-NEXT: [[d0:%.*]] = call <4 x i32> @llvm.zluda.mma.m16n8k16.s32.s8.s8.s32(<2 x i32> [[a0]], <1 x i32> [[b0]], <4 x i32> %c)
; CHECK-NEXT: [[d1:%.*]] = call <4 x i32> @llvm.zluda.mma.m16n8k16.s32.s8.s8.s32(<2 x i32> [[a1]], <1 x i32> [[b1]], <4 x i32> [[d0]])
; CHECK-NEXT: store <4 x i32> [[d1]], ptr %d.result, align 16
; CHECK-NEXT: ret void
define void @split_combine(ptr %d0.result, ptr %d1.result, <4 x i32> %a0, <2 x i32> %b0, <4 x i32> %c0, <4 x i32> %a1, <2 x i32> %b1, <4 x i32> %c1) {
  %d0 = call <4 x i32> @llvm.zluda.mma.m16n8k32.s32.s8.s8.s32(<4 x i32> %a0, <2 x i32> %b0, <4 x i32> %c0)
  %d1 = call <4 x i32> @llvm.zluda.mma.m16n8k32.s32.s8.s8.s32(<4 x i32> %a1, <2 x i32> %b1, <4 x i32> %c1)
  store <4 x i32> %d0, ptr %d0.result
  store <4 x i32> %d1, ptr %d1.result
  ret void
}

declare <4 x i32> @llvm.zluda.mma.m16n8k32.s32.s8.s8.s32(<4 x i32>, <2 x i32>, <4 x i32>)