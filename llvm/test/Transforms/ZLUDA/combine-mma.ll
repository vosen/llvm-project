; RUN: opt -S -passes=zluda-combine-mma -mtriple=amdgcn-amd-amdhsa %s | FileCheck %s

; CHECK-LABEL: @single_call
; CHECK-NEXT: [[a_shuffle:%.*]] = call <16 x i16> @llvm.zluda.amatrix.convert.amd.nv16x16.v16i16.v4i32(<4 x i32> %a)
; CHECK-NEXT: [[b_zext:%.*]] = call <16 x i16> @llvm.zluda.bmatrix.zext.amd16x16.nv16x8.v16i16.v2i32(<2 x i32> %b)
; CHECK-NEXT: [[c_zext:%.*]] = call <8 x float> @llvm.zluda.cmatrix.zext.amd16x16.nv16x8.v8f32.v4f32(<4 x float> %c)
; CHECK-NEXT: [[d:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> [[a_shuffle]], <16 x i16> [[b_zext]], <8 x float> [[c_zext]])
; CHECK-NEXT: [[d_result:%.*]] = call <4 x float> @llvm.zluda.dmatrix.trunc.nv16x8.amd16x16(<8 x float> [[d]])
; CHECK-NEXT: store <4 x float> [[d_result]], ptr %d.result, align 16
; CHECK-NEXT: ret void
define void @single_call(ptr %d.result, <4 x i32> %a, <2 x i32> %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %b, <4 x float> %c)
  store <4 x float> %d, ptr %d.result
  ret void
}

; CHECK-LABEL: @no_shared_a
; CHECK: [[d0:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: [[d0_result:%.*]] = call <4 x float> @llvm.zluda.dmatrix.trunc.nv16x8.amd16x16(<8 x float> [[d0]])
; CHECK: [[d1:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: [[d1_result:%.*]] = call <4 x float> @llvm.zluda.dmatrix.trunc.nv16x8.amd16x16(<8 x float> [[d1]])
; CHECK: store <4 x float> [[d0_result]], ptr %d0.result, align 16
; CHECK: store <4 x float> [[d1_result]], ptr %d1.result, align 16
define void @no_shared_a(ptr %d0.result, ptr %d1.result, <4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0, <4 x i32> %a1, <2 x i32> %b1, <4 x float> %c1) {
  %d0 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0)
  %d1 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a1, <2 x i32> %b1, <4 x float> %c1)

  store <4 x float> %d0, ptr %d0.result
  store <4 x float> %d1, ptr %d1.result

  ret void
}

; CHECK-LABEL: @dependency
; CHECK: [[d0:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: [[d0_result:%.*]] = call <4 x float> @llvm.zluda.dmatrix.trunc.nv16x8.amd16x16(<8 x float> [[d0]])
; CHECK: [[d1:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: [[d1_result:%.*]] = call <4 x float> @llvm.zluda.dmatrix.trunc.nv16x8.amd16x16(<8 x float> [[d1]])
; CHECK: store <4 x float> [[d0_result]], ptr %d0.result, align 16
; CHECK: store <4 x float> [[d1_result]], ptr %d1.result, align 16
define void @dependency(ptr %d0.result, ptr %d1.result, <4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0, <2 x i32> %b1) {
  %d0 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0)
  %d1 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b1, <4 x float> %d0)

  store <4 x float> %d0, ptr %d0.result
  store <4 x float> %d1, ptr %d1.result

  ret void
}

; CHECK-LABEL: @zext_trunc_elimination
; CHECK: [[first:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: {{%.*}} = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> [[first]])
define void @zext_trunc_elimination(ptr %d0.result, ptr %d1.result, <4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0, <2 x i32> %b1) {
  %d0 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0)
  %d1 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b1, <4 x float> %d0)

  store <4 x float> %d0, ptr %d0.result
  store <4 x float> %d1, ptr %d1.result

  ret void
}

; CHECK-LABEL: @combine
; CHECK-NEXT: [[shuffled_a:%.*]] = call <16 x i16> @llvm.zluda.amatrix.convert.amd.nv16x16.v16i16.v4i32(<4 x i32> %a)
; CHECK-NEXT: [[combined_b:%.*]] = call <16 x i16> @llvm.zluda.bmatrix.concatenate.amd16x16.nv16x8.v16i16.v2i32(<2 x i32> %b0, <2 x i32> %b1)
; CHECK-NEXT: [[combined_c:%.*]] = call <8 x float> @llvm.zluda.cmatrix.concatenate.amd16x16.nv16x8.v8f32.v4f32(<4 x float> %c0, <4 x float> %c1)
; CHECK-NEXT: [[result:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> [[shuffled_a]], <16 x i16> [[combined_b]], <8 x float> [[combined_c]])
; CHECK-NEXT: [[split:%.*]] = call { <4 x float>, <4 x float> } @llvm.zluda.dmatrix.split.nv16x8.amd16x16(<8 x float> [[result]])
; CHECK-NEXT: [[d0:%.*]] = extractvalue { <4 x float>, <4 x float> } [[split]], 0
; CHECK-NEXT: [[d1:%.*]] = extractvalue { <4 x float>, <4 x float> } [[split]], 1
; CHECK-NEXT: store <4 x float> [[d0]], ptr %d0.result, align 16
; CHECK-NEXT: store <4 x float> [[d1]], ptr %d1.result, align 16
; CHECK-NEXT: ret void
define void @combine(ptr %d0.result, ptr %d1.result, <4 x i32> %a, <2 x i32> %b0, <4 x float> %c0, <2 x i32> %b1, <4 x float> %c1) {
  %d0 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %b0, <4 x float> %c0)
  %d1 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %b1, <4 x float> %c1)

  store <4 x float> %d0, ptr %d0.result
  store <4 x float> %d1, ptr %d1.result

  ret void
}

; CHECK-LABEL: @use_in_middle
; CHECK: [[result:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: [[split:%.*]] = call { <4 x float>, <4 x float> } @llvm.zluda.dmatrix.split.nv16x8.amd16x16(<8 x float> [[result]])
; CHECK: [[d0:%.*]] = extractvalue { <4 x float>, <4 x float> } [[split]], 0
; CHECK: [[d1:%.*]] = extractvalue { <4 x float>, <4 x float> } [[split]], 1
; CHECK: {{%.*}} = extractelement <4 x float> [[d0]], i64 0
define void @use_in_middle(ptr %d0.result, ptr %d1.result, <4 x i32> %a, <2 x i32> %b0, <4 x float> %c0, <2 x i32> %b1, <4 x float> %c1) {
  %d0 = tail call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %b0, <4 x float> %c0)
  %2 = extractelement <4 x float> %d0, i64 0
  %d1 = tail call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %b1, <4 x float> %c1)
  store <4 x float> %d0, ptr %d0.result
  store <4 x float> %d1, ptr %d1.result

  ret void
}

; CHECK-LABEL: @dependency_in_middle
; CHECK: %dependency = insertelement <2 x i32> zeroinitializer, i32 0, i64 0
; CHECK: [[combined_b:%.*]] = call <16 x i16> @llvm.zluda.bmatrix.concatenate.amd16x16.nv16x8.v16i16.v2i32(<2 x i32> %b0, <2 x i32> %dependency)
; CHECK: [[result:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> [[combined_b]], <8 x float> {{%.*}})
define void @dependency_in_middle(ptr %d0.result, ptr %d1.result, <4 x i32> %a, <2 x i32> %b0, <4 x float> %c0, <4 x float> %c1) {
  %d0 = tail call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %b0, <4 x float> %c0)
  %dependency = insertelement <2 x i32> zeroinitializer, i32 0, i64 0
  %d1 = tail call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a, <2 x i32> %dependency, <4 x float> %c1)
  store <4 x float> %d0, ptr %d0.result
  store <4 x float> %d1, ptr %d1.result

  ret void
}

; CHECK-LABEL: @split_combine_elimination
; CHECK: [[first:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> {{%.*}})
; CHECK: {{%.*}} = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> {{%.*}}, <16 x i16> {{%.*}}, <8 x float> [[first]])
define void @split_combine_elimination(ptr %d2.result, ptr %d3.result, <4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0, <2 x i32> %b1, <4 x float> %c1, <4 x i32> %a1, <2 x i32> %b2, <2 x i32> %b3) {
  %d0 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b0, <4 x float> %c0)
  %d1 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a0, <2 x i32> %b1, <4 x float> %c1)

  %d2 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a1, <2 x i32> %b2, <4 x float> %d0)
  %d3 = call <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32> %a1, <2 x i32> %b3, <4 x float> %d1)
  store <4 x float> %d2, ptr %d2.result
  store <4 x float> %d3, ptr %d3.result

  ret void
}

; CHECK-LABEL: @combine_s32
; CHECK-NEXT: [[shuffled_a:%.*]] = call <2 x i32> @llvm.zluda.amatrix.convert.amd.nv16x16.v16i16.v4i32(<4 x i32> %a)
; CHECK-NEXT: [[combined_b:%.*]] = call <2 x i32> @llvm.zluda.bmatrix.concatenate.amd16x16.nv16x8.v16i16.v2i32(<2 x i32> %b0, <2 x i32> %b1)
; CHECK-NEXT: [[combined_c:%.*]] = call <8 x i32> @llvm.zluda.cmatrix.concatenate.amd16x16.nv16x8.v8f32.v4f32(<4 x float> %c0, <4 x float> %c1)
; CHECK-NEXT: [[result:%.*]] = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8f32.v16i16(i1 true, <2 x i32> [[shuffled_a]], i1 true, <2 x i32> [[combined_b]], <8 x i32> [[combined_c]], i1 false)
; CHECK-NEXT: [[split:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.zluda.dmatrix.split.nv16x8.amd16x16(<8 x i32> [[result]])
; CHECK-NEXT: [[d0:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[split]], 0
; CHECK-NEXT: [[d1:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[split]], 1
; CHECK-NEXT: store <4 x i32> [[d0]], ptr %d0.result, align 16
; CHECK-NEXT: store <4 x i32> [[d1]], ptr %d1.result, align 16
; CHECK-NEXT: ret void
define void @combine_s32(ptr %d0.result, ptr %d1.result, <2 x i32> %a, <1 x i32> %b0, <4 x i32> %c0, <1 x i32> %b1, <4 x i32> %c1) {
  %d0 = call <4 x i32> @llvm.zluda.mma.m16n8k16.s32.s8.s8.s32(<2 x i32> %a, <1 x i32> %b0, <4 x i32> %c0)
  %d1 = call <4 x i32> @llvm.zluda.mma.m16n8k16.s32.s8.s8.s32(<2 x i32> %a, <1 x i32> %b1, <4 x i32> %c1)

  store <4 x i32> %d0, ptr %d0.result
  store <4 x i32> %d1, ptr %d1.result

  ret void
}

declare <4 x float> @llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32(<4 x i32>, <2 x i32>, <4 x float>)
