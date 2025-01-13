// ZLUDA-specific generic intrinsics. Should be removed by lowering passes before reaching codegen.

def int_zluda_amatrix_convert_amd_nv16x16 : DefaultAttrsIntrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty],
  [IntrNoMem, IntrConvergent]
>;

def int_zluda_bmatrix_zext_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_cmatrix_zext_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_dmatrix_trunc_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4f32_ty], [llvm_v8f32_ty],
  [IntrNoMem, IntrConvergent]
>;

def int_zluda_bmatrix_concatenate_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty, LLVMMatchType<1>],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_cmatrix_concatenate_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty, LLVMMatchType<1>],
  [IntrNoMem, IntrConvergent]
>;

def int_zluda_dmatrix_split_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4f32_ty, llvm_v4f32_ty], [llvm_v8f32_ty],
  [IntrNoMem, IntrConvergent]
>;

def int_zluda_mma_m16n8k16 : DefaultAttrsIntrinsic<
  [llvm_v4f32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4f32_ty],
  [IntrNoMem, IntrConvergent]
>;
