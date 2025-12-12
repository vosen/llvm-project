// ZLUDA-specific generic intrinsics. Should be removed by lowering passes before reaching codegen.

// Input intrinsics, this is what ZLUDA emits, all of them use NV hardware
// layout as described in PTX documentation. None of them are lowered to
// AMD hardware instructions

// The return type should be llvm_v4f32_ty to match the PTX definition,
// but if the return type is llvm_v4i32_ty it's easier to perform
// D matrix reconstitution
def int_zluda_mma_m16n8k16_f32_bf16_bf16_f32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_mma_m16n8k16_f32_f16_f16_f32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_mma_m16n8k32_s32_s8_s8_s32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// Successful matrix layout change. We convert inputs and split outputs

// C matrix and D matrix conversion is shared between all fp16,bf16, I8
def int_zluda_cmatrix_concatenate_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_dmatrix_split_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v8i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_dmatrix_trunc_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v8f32_ty],
  [IntrNoMem, IntrConvergent]
>;

// BF16/FP16
def int_zluda_amatrix_convert_amd_nv16x16 : DefaultAttrsIntrinsic<
  [llvm_v16i16_ty], [llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_bmatrix_concatenate_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_v16i16_ty], [llvm_v2i32_ty, llvm_v2i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// I8
def int_zluda_amatrix_split_amd16x16_nv16x32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_bmatrix_reshape_amd16x16_nv32x8 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// TODO: custom classes to reduce code duplication in ZLUDA intrinsics
