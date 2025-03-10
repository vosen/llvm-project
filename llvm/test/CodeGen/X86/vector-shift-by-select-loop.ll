; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+xop | FileCheck %s --check-prefix=XOP

; PR37428 - https://bugs.llvm.org/show_bug.cgi?id=37428
; This is a larger-than-usual regression test to verify that several backend
; transforms are working together. We want to hoist the expansion of non-uniform
; vector shifts out of a loop if we do not have real vector shift instructions.
; See test/Transforms/CodeGenPrepare/X86/vec-shift.ll for the 1st step in that
; sequence.

define void @vector_variable_shift_left_loop(i32* nocapture %arr, i8* nocapture readonly %control, i32 %count, i32 %amt0, i32 %amt1) nounwind {
; SSE-LABEL: vector_variable_shift_left_loop:
; SSE:       # %bb.0: # %entry
; SSE-NEXT:    testl %edx, %edx
; SSE-NEXT:    jle .LBB0_9
; SSE-NEXT:  # %bb.1: # %for.body.preheader
; SSE-NEXT:    movl %ecx, %eax
; SSE-NEXT:    movl %edx, %r9d
; SSE-NEXT:    cmpl $31, %edx
; SSE-NEXT:    ja .LBB0_3
; SSE-NEXT:  # %bb.2:
; SSE-NEXT:    xorl %edx, %edx
; SSE-NEXT:    jmp .LBB0_6
; SSE-NEXT:  .LBB0_3: # %vector.ph
; SSE-NEXT:    movl %r9d, %edx
; SSE-NEXT:    andl $-32, %edx
; SSE-NEXT:    movd %eax, %xmm0
; SSE-NEXT:    movd %r8d, %xmm1
; SSE-NEXT:    xorl %ecx, %ecx
; SSE-NEXT:    pxor %xmm8, %xmm8
; SSE-NEXT:    pmovzxdq {{.*#+}} xmm9 = xmm0[0],zero,xmm0[1],zero
; SSE-NEXT:    pmovzxdq {{.*#+}} xmm10 = xmm1[0],zero,xmm1[1],zero
; SSE-NEXT:    .p2align 4, 0x90
; SSE-NEXT:  .LBB0_4: # %vector.body
; SSE-NEXT:    # =>This Inner Loop Header: Depth=1
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE-NEXT:    movq {{.*#+}} xmm2 = mem[0],zero
; SSE-NEXT:    movq {{.*#+}} xmm11 = mem[0],zero
; SSE-NEXT:    pcmpeqb %xmm8, %xmm0
; SSE-NEXT:    pmovsxbd %xmm0, %xmm7
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,1,1,1]
; SSE-NEXT:    pmovsxbd %xmm0, %xmm0
; SSE-NEXT:    pcmpeqb %xmm8, %xmm1
; SSE-NEXT:    pmovsxbd %xmm1, %xmm5
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,1,1]
; SSE-NEXT:    pmovsxbd %xmm1, %xmm6
; SSE-NEXT:    pcmpeqb %xmm8, %xmm2
; SSE-NEXT:    pmovsxbd %xmm2, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm2[1,1,1,1]
; SSE-NEXT:    pmovsxbd %xmm1, %xmm4
; SSE-NEXT:    pcmpeqb %xmm8, %xmm11
; SSE-NEXT:    pmovsxbd %xmm11, %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm11[1,1,1,1]
; SSE-NEXT:    pmovsxbd %xmm2, %xmm2
; SSE-NEXT:    movdqu 16(%rdi,%rcx,4), %xmm11
; SSE-NEXT:    movdqa %xmm11, %xmm12
; SSE-NEXT:    pslld %xmm9, %xmm12
; SSE-NEXT:    pslld %xmm10, %xmm11
; SSE-NEXT:    blendvps %xmm0, %xmm12, %xmm11
; SSE-NEXT:    movdqu (%rdi,%rcx,4), %xmm12
; SSE-NEXT:    movdqa %xmm12, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm12
; SSE-NEXT:    movdqa %xmm7, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm12
; SSE-NEXT:    movdqu 48(%rdi,%rcx,4), %xmm7
; SSE-NEXT:    movdqa %xmm7, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm7
; SSE-NEXT:    movdqa %xmm6, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm7
; SSE-NEXT:    movdqu 32(%rdi,%rcx,4), %xmm6
; SSE-NEXT:    movdqa %xmm6, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm6
; SSE-NEXT:    movdqa %xmm5, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm6
; SSE-NEXT:    movdqu 80(%rdi,%rcx,4), %xmm5
; SSE-NEXT:    movdqa %xmm5, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm5
; SSE-NEXT:    movdqa %xmm4, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm5
; SSE-NEXT:    movdqu 64(%rdi,%rcx,4), %xmm4
; SSE-NEXT:    movdqa %xmm4, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm4
; SSE-NEXT:    movdqa %xmm3, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm4
; SSE-NEXT:    movdqu 112(%rdi,%rcx,4), %xmm3
; SSE-NEXT:    movdqa %xmm3, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm3
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm3
; SSE-NEXT:    movdqu 96(%rdi,%rcx,4), %xmm2
; SSE-NEXT:    movdqa %xmm2, %xmm13
; SSE-NEXT:    pslld %xmm9, %xmm13
; SSE-NEXT:    pslld %xmm10, %xmm2
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    blendvps %xmm0, %xmm13, %xmm2
; SSE-NEXT:    movups %xmm12, (%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm11, 16(%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm6, 32(%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm7, 48(%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm4, 64(%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm5, 80(%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm2, 96(%rdi,%rcx,4)
; SSE-NEXT:    movups %xmm3, 112(%rdi,%rcx,4)
; SSE-NEXT:    addq $32, %rcx
; SSE-NEXT:    cmpq %rcx, %rdx
; SSE-NEXT:    jne .LBB0_4
; SSE-NEXT:  # %bb.5: # %middle.block
; SSE-NEXT:    cmpl %r9d, %edx
; SSE-NEXT:    jne .LBB0_6
; SSE-NEXT:  .LBB0_9: # %for.cond.cleanup
; SSE-NEXT:    retq
; SSE-NEXT:    .p2align 4, 0x90
; SSE-NEXT:  .LBB0_8: # %for.body
; SSE-NEXT:    # in Loop: Header=BB0_6 Depth=1
; SSE-NEXT:    # kill: def $cl killed $cl killed $ecx
; SSE-NEXT:    shll %cl, (%rdi,%rdx,4)
; SSE-NEXT:    incq %rdx
; SSE-NEXT:    cmpq %rdx, %r9
; SSE-NEXT:    je .LBB0_9
; SSE-NEXT:  .LBB0_6: # %for.body
; SSE-NEXT:    # =>This Inner Loop Header: Depth=1
; SSE-NEXT:    cmpb $0, (%rsi,%rdx)
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    je .LBB0_8
; SSE-NEXT:  # %bb.7: # %for.body
; SSE-NEXT:    # in Loop: Header=BB0_6 Depth=1
; SSE-NEXT:    movl %r8d, %ecx
; SSE-NEXT:    jmp .LBB0_8
;
; AVX1-LABEL: vector_variable_shift_left_loop:
; AVX1:       # %bb.0: # %entry
; AVX1-NEXT:    testl %edx, %edx
; AVX1-NEXT:    jle .LBB0_9
; AVX1-NEXT:  # %bb.1: # %for.body.preheader
; AVX1-NEXT:    movl %ecx, %eax
; AVX1-NEXT:    movl %edx, %r9d
; AVX1-NEXT:    cmpl $31, %edx
; AVX1-NEXT:    ja .LBB0_3
; AVX1-NEXT:  # %bb.2:
; AVX1-NEXT:    xorl %edx, %edx
; AVX1-NEXT:    jmp .LBB0_6
; AVX1-NEXT:  .LBB0_3: # %vector.ph
; AVX1-NEXT:    movl %r9d, %edx
; AVX1-NEXT:    andl $-32, %edx
; AVX1-NEXT:    vmovd %eax, %xmm7
; AVX1-NEXT:    vmovd %r8d, %xmm8
; AVX1-NEXT:    xorl %ecx, %ecx
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm7[0],zero,xmm7[1],zero
; AVX1-NEXT:    vmovdqa %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm8[0],zero,xmm8[1],zero
; AVX1-NEXT:    vmovdqa %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm7[0],zero,xmm7[1],zero
; AVX1-NEXT:    vmovdqa %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm8[0],zero,xmm8[1],zero
; AVX1-NEXT:    vmovdqa %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm5 = xmm7[0],zero,xmm7[1],zero
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm6 = xmm8[0],zero,xmm8[1],zero
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm7 = xmm7[0],zero,xmm7[1],zero
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm8 = xmm8[0],zero,xmm8[1],zero
; AVX1-NEXT:    .p2align 4, 0x90
; AVX1-NEXT:  .LBB0_4: # %vector.body
; AVX1-NEXT:    # =>This Inner Loop Header: Depth=1
; AVX1-NEXT:    vmovq {{.*#+}} xmm9 = mem[0],zero
; AVX1-NEXT:    vmovq {{.*#+}} xmm10 = mem[0],zero
; AVX1-NEXT:    vmovq {{.*#+}} xmm11 = mem[0],zero
; AVX1-NEXT:    vmovq {{.*#+}} xmm12 = mem[0],zero
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpeqb %xmm2, %xmm9, %xmm9
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm14
; AVX1-NEXT:    vpshufd {{.*#+}} xmm9 = xmm9[1,1,1,1]
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm15
; AVX1-NEXT:    vpcmpeqb %xmm2, %xmm10, %xmm9
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm0
; AVX1-NEXT:    vpshufd {{.*#+}} xmm9 = xmm9[1,1,1,1]
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm1
; AVX1-NEXT:    vpcmpeqb %xmm2, %xmm11, %xmm9
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm13
; AVX1-NEXT:    vpshufd {{.*#+}} xmm9 = xmm9[1,1,1,1]
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm11
; AVX1-NEXT:    vpcmpeqb %xmm2, %xmm12, %xmm9
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm10
; AVX1-NEXT:    vpshufd {{.*#+}} xmm9 = xmm9[1,1,1,1]
; AVX1-NEXT:    vpmovsxbd %xmm9, %xmm9
; AVX1-NEXT:    vmovdqu (%rdi,%rcx,4), %xmm12
; AVX1-NEXT:    vmovdqa {{[-0-9]+}}(%r{{[sb]}}p), %xmm3 # 16-byte Reload
; AVX1-NEXT:    vpslld %xmm3, %xmm12, %xmm2
; AVX1-NEXT:    vmovdqa {{[-0-9]+}}(%r{{[sb]}}p), %xmm4 # 16-byte Reload
; AVX1-NEXT:    vpslld %xmm4, %xmm12, %xmm12
; AVX1-NEXT:    vblendvps %xmm14, %xmm2, %xmm12, %xmm12
; AVX1-NEXT:    vmovdqu 16(%rdi,%rcx,4), %xmm2
; AVX1-NEXT:    vpslld %xmm3, %xmm2, %xmm14
; AVX1-NEXT:    vpslld %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vblendvps %xmm15, %xmm14, %xmm2, %xmm2
; AVX1-NEXT:    vmovdqu 32(%rdi,%rcx,4), %xmm14
; AVX1-NEXT:    vmovdqa {{[-0-9]+}}(%r{{[sb]}}p), %xmm3 # 16-byte Reload
; AVX1-NEXT:    vpslld %xmm3, %xmm14, %xmm15
; AVX1-NEXT:    vmovdqa {{[-0-9]+}}(%r{{[sb]}}p), %xmm4 # 16-byte Reload
; AVX1-NEXT:    vpslld %xmm4, %xmm14, %xmm14
; AVX1-NEXT:    vblendvps %xmm0, %xmm15, %xmm14, %xmm0
; AVX1-NEXT:    vmovdqu 48(%rdi,%rcx,4), %xmm14
; AVX1-NEXT:    vpslld %xmm3, %xmm14, %xmm15
; AVX1-NEXT:    vpslld %xmm4, %xmm14, %xmm14
; AVX1-NEXT:    vblendvps %xmm1, %xmm15, %xmm14, %xmm1
; AVX1-NEXT:    vmovdqu 64(%rdi,%rcx,4), %xmm14
; AVX1-NEXT:    vpslld %xmm5, %xmm14, %xmm15
; AVX1-NEXT:    vpslld %xmm6, %xmm14, %xmm14
; AVX1-NEXT:    vblendvps %xmm13, %xmm15, %xmm14, %xmm13
; AVX1-NEXT:    vmovdqu 80(%rdi,%rcx,4), %xmm14
; AVX1-NEXT:    vpslld %xmm5, %xmm14, %xmm15
; AVX1-NEXT:    vpslld %xmm6, %xmm14, %xmm14
; AVX1-NEXT:    vblendvps %xmm11, %xmm15, %xmm14, %xmm11
; AVX1-NEXT:    vmovdqu 96(%rdi,%rcx,4), %xmm14
; AVX1-NEXT:    vpslld %xmm7, %xmm14, %xmm15
; AVX1-NEXT:    vpslld %xmm8, %xmm14, %xmm14
; AVX1-NEXT:    vblendvps %xmm10, %xmm15, %xmm14, %xmm10
; AVX1-NEXT:    vmovdqu 112(%rdi,%rcx,4), %xmm14
; AVX1-NEXT:    vpslld %xmm7, %xmm14, %xmm15
; AVX1-NEXT:    vpslld %xmm8, %xmm14, %xmm14
; AVX1-NEXT:    vblendvps %xmm9, %xmm15, %xmm14, %xmm9
; AVX1-NEXT:    vmovups %xmm12, (%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm2, 16(%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm0, 32(%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm1, 48(%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm13, 64(%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm11, 80(%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm10, 96(%rdi,%rcx,4)
; AVX1-NEXT:    vmovups %xmm9, 112(%rdi,%rcx,4)
; AVX1-NEXT:    addq $32, %rcx
; AVX1-NEXT:    cmpq %rcx, %rdx
; AVX1-NEXT:    jne .LBB0_4
; AVX1-NEXT:  # %bb.5: # %middle.block
; AVX1-NEXT:    cmpl %r9d, %edx
; AVX1-NEXT:    jne .LBB0_6
; AVX1-NEXT:  .LBB0_9: # %for.cond.cleanup
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
; AVX1-NEXT:    .p2align 4, 0x90
; AVX1-NEXT:  .LBB0_8: # %for.body
; AVX1-NEXT:    # in Loop: Header=BB0_6 Depth=1
; AVX1-NEXT:    # kill: def $cl killed $cl killed $ecx
; AVX1-NEXT:    shll %cl, (%rdi,%rdx,4)
; AVX1-NEXT:    incq %rdx
; AVX1-NEXT:    cmpq %rdx, %r9
; AVX1-NEXT:    je .LBB0_9
; AVX1-NEXT:  .LBB0_6: # %for.body
; AVX1-NEXT:    # =>This Inner Loop Header: Depth=1
; AVX1-NEXT:    cmpb $0, (%rsi,%rdx)
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    je .LBB0_8
; AVX1-NEXT:  # %bb.7: # %for.body
; AVX1-NEXT:    # in Loop: Header=BB0_6 Depth=1
; AVX1-NEXT:    movl %r8d, %ecx
; AVX1-NEXT:    jmp .LBB0_8
;
; AVX2-LABEL: vector_variable_shift_left_loop:
; AVX2:       # %bb.0: # %entry
; AVX2-NEXT:    testl %edx, %edx
; AVX2-NEXT:    jle .LBB0_9
; AVX2-NEXT:  # %bb.1: # %for.body.preheader
; AVX2-NEXT:    movl %ecx, %eax
; AVX2-NEXT:    movl %edx, %r9d
; AVX2-NEXT:    cmpl $31, %edx
; AVX2-NEXT:    ja .LBB0_3
; AVX2-NEXT:  # %bb.2:
; AVX2-NEXT:    xorl %edx, %edx
; AVX2-NEXT:    jmp .LBB0_6
; AVX2-NEXT:  .LBB0_3: # %vector.ph
; AVX2-NEXT:    movl %r9d, %edx
; AVX2-NEXT:    andl $-32, %edx
; AVX2-NEXT:    vmovd %eax, %xmm0
; AVX2-NEXT:    vpbroadcastd %xmm0, %ymm0
; AVX2-NEXT:    vmovd %r8d, %xmm1
; AVX2-NEXT:    vpbroadcastd %xmm1, %ymm1
; AVX2-NEXT:    xorl %ecx, %ecx
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    .p2align 4, 0x90
; AVX2-NEXT:  .LBB0_4: # %vector.body
; AVX2-NEXT:    # =>This Inner Loop Header: Depth=1
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm3 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero,mem[4],zero,zero,zero,mem[5],zero,zero,zero,mem[6],zero,zero,zero,mem[7],zero,zero,zero
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm4 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero,mem[4],zero,zero,zero,mem[5],zero,zero,zero,mem[6],zero,zero,zero,mem[7],zero,zero,zero
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm5 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero,mem[4],zero,zero,zero,mem[5],zero,zero,zero,mem[6],zero,zero,zero,mem[7],zero,zero,zero
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm6 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero,mem[4],zero,zero,zero,mem[5],zero,zero,zero,mem[6],zero,zero,zero,mem[7],zero,zero,zero
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm3, %ymm3
; AVX2-NEXT:    vblendvps %ymm3, %ymm0, %ymm1, %ymm3
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm4, %ymm4
; AVX2-NEXT:    vblendvps %ymm4, %ymm0, %ymm1, %ymm4
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm5, %ymm5
; AVX2-NEXT:    vblendvps %ymm5, %ymm0, %ymm1, %ymm5
; AVX2-NEXT:    vpcmpeqd %ymm2, %ymm6, %ymm6
; AVX2-NEXT:    vblendvps %ymm6, %ymm0, %ymm1, %ymm6
; AVX2-NEXT:    vmovdqu (%rdi,%rcx,4), %ymm7
; AVX2-NEXT:    vpsllvd %ymm3, %ymm7, %ymm3
; AVX2-NEXT:    vmovdqu 32(%rdi,%rcx,4), %ymm7
; AVX2-NEXT:    vpsllvd %ymm4, %ymm7, %ymm4
; AVX2-NEXT:    vmovdqu 64(%rdi,%rcx,4), %ymm7
; AVX2-NEXT:    vpsllvd %ymm5, %ymm7, %ymm5
; AVX2-NEXT:    vmovdqu 96(%rdi,%rcx,4), %ymm7
; AVX2-NEXT:    vpsllvd %ymm6, %ymm7, %ymm6
; AVX2-NEXT:    vmovdqu %ymm3, (%rdi,%rcx,4)
; AVX2-NEXT:    vmovdqu %ymm4, 32(%rdi,%rcx,4)
; AVX2-NEXT:    vmovdqu %ymm5, 64(%rdi,%rcx,4)
; AVX2-NEXT:    vmovdqu %ymm6, 96(%rdi,%rcx,4)
; AVX2-NEXT:    addq $32, %rcx
; AVX2-NEXT:    cmpq %rcx, %rdx
; AVX2-NEXT:    jne .LBB0_4
; AVX2-NEXT:  # %bb.5: # %middle.block
; AVX2-NEXT:    cmpl %r9d, %edx
; AVX2-NEXT:    jne .LBB0_6
; AVX2-NEXT:  .LBB0_9: # %for.cond.cleanup
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
; AVX2-NEXT:    .p2align 4, 0x90
; AVX2-NEXT:  .LBB0_8: # %for.body
; AVX2-NEXT:    # in Loop: Header=BB0_6 Depth=1
; AVX2-NEXT:    # kill: def $cl killed $cl killed $ecx
; AVX2-NEXT:    shll %cl, (%rdi,%rdx,4)
; AVX2-NEXT:    incq %rdx
; AVX2-NEXT:    cmpq %rdx, %r9
; AVX2-NEXT:    je .LBB0_9
; AVX2-NEXT:  .LBB0_6: # %for.body
; AVX2-NEXT:    # =>This Inner Loop Header: Depth=1
; AVX2-NEXT:    cmpb $0, (%rsi,%rdx)
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    je .LBB0_8
; AVX2-NEXT:  # %bb.7: # %for.body
; AVX2-NEXT:    # in Loop: Header=BB0_6 Depth=1
; AVX2-NEXT:    movl %r8d, %ecx
; AVX2-NEXT:    jmp .LBB0_8
;
; XOP-LABEL: vector_variable_shift_left_loop:
; XOP:       # %bb.0: # %entry
; XOP-NEXT:    testl %edx, %edx
; XOP-NEXT:    jle .LBB0_9
; XOP-NEXT:  # %bb.1: # %for.body.preheader
; XOP-NEXT:    movl %ecx, %eax
; XOP-NEXT:    movl %edx, %r9d
; XOP-NEXT:    cmpl $31, %edx
; XOP-NEXT:    ja .LBB0_3
; XOP-NEXT:  # %bb.2:
; XOP-NEXT:    xorl %edx, %edx
; XOP-NEXT:    jmp .LBB0_6
; XOP-NEXT:  .LBB0_3: # %vector.ph
; XOP-NEXT:    movl %r9d, %edx
; XOP-NEXT:    andl $-32, %edx
; XOP-NEXT:    vmovd %eax, %xmm0
; XOP-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,0,0,0]
; XOP-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; XOP-NEXT:    vmovd %r8d, %xmm1
; XOP-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,0,0,0]
; XOP-NEXT:    vinsertf128 $1, %xmm1, %ymm1, %ymm1
; XOP-NEXT:    xorl %ecx, %ecx
; XOP-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; XOP-NEXT:    vextractf128 $1, %ymm0, %xmm3
; XOP-NEXT:    vextractf128 $1, %ymm1, %xmm4
; XOP-NEXT:    .p2align 4, 0x90
; XOP-NEXT:  .LBB0_4: # %vector.body
; XOP-NEXT:    # =>This Inner Loop Header: Depth=1
; XOP-NEXT:    vmovq {{.*#+}} xmm5 = mem[0],zero
; XOP-NEXT:    vmovq {{.*#+}} xmm6 = mem[0],zero
; XOP-NEXT:    vmovq {{.*#+}} xmm7 = mem[0],zero
; XOP-NEXT:    vmovq {{.*#+}} xmm8 = mem[0],zero
; XOP-NEXT:    vpcomeqb %xmm2, %xmm5, %xmm5
; XOP-NEXT:    vpmovsxbd %xmm5, %xmm9
; XOP-NEXT:    vpshufd {{.*#+}} xmm5 = xmm5[1,1,1,1]
; XOP-NEXT:    vpmovsxbd %xmm5, %xmm5
; XOP-NEXT:    vpcomeqb %xmm2, %xmm6, %xmm6
; XOP-NEXT:    vpmovsxbd %xmm6, %xmm10
; XOP-NEXT:    vpshufd {{.*#+}} xmm6 = xmm6[1,1,1,1]
; XOP-NEXT:    vpmovsxbd %xmm6, %xmm6
; XOP-NEXT:    vpcomeqb %xmm2, %xmm7, %xmm7
; XOP-NEXT:    vpmovsxbd %xmm7, %xmm11
; XOP-NEXT:    vpshufd {{.*#+}} xmm7 = xmm7[1,1,1,1]
; XOP-NEXT:    vpmovsxbd %xmm7, %xmm7
; XOP-NEXT:    vpcomeqb %xmm2, %xmm8, %xmm8
; XOP-NEXT:    vpmovsxbd %xmm8, %xmm12
; XOP-NEXT:    vpshufd {{.*#+}} xmm8 = xmm8[1,1,1,1]
; XOP-NEXT:    vpmovsxbd %xmm8, %xmm8
; XOP-NEXT:    vblendvps %xmm5, %xmm3, %xmm4, %xmm5
; XOP-NEXT:    vpshld %xmm5, 16(%rdi,%rcx,4), %xmm5
; XOP-NEXT:    vblendvps %xmm9, %xmm0, %xmm1, %xmm9
; XOP-NEXT:    vpshld %xmm9, (%rdi,%rcx,4), %xmm9
; XOP-NEXT:    vblendvps %xmm6, %xmm3, %xmm4, %xmm6
; XOP-NEXT:    vpshld %xmm6, 48(%rdi,%rcx,4), %xmm6
; XOP-NEXT:    vblendvps %xmm10, %xmm0, %xmm1, %xmm10
; XOP-NEXT:    vpshld %xmm10, 32(%rdi,%rcx,4), %xmm10
; XOP-NEXT:    vblendvps %xmm7, %xmm3, %xmm4, %xmm7
; XOP-NEXT:    vpshld %xmm7, 80(%rdi,%rcx,4), %xmm7
; XOP-NEXT:    vblendvps %xmm11, %xmm0, %xmm1, %xmm11
; XOP-NEXT:    vpshld %xmm11, 64(%rdi,%rcx,4), %xmm11
; XOP-NEXT:    vblendvps %xmm8, %xmm3, %xmm4, %xmm8
; XOP-NEXT:    vpshld %xmm8, 112(%rdi,%rcx,4), %xmm8
; XOP-NEXT:    vblendvps %xmm12, %xmm0, %xmm1, %xmm12
; XOP-NEXT:    vpshld %xmm12, 96(%rdi,%rcx,4), %xmm12
; XOP-NEXT:    vmovdqu %xmm9, (%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm5, 16(%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm10, 32(%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm6, 48(%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm11, 64(%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm7, 80(%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm12, 96(%rdi,%rcx,4)
; XOP-NEXT:    vmovdqu %xmm8, 112(%rdi,%rcx,4)
; XOP-NEXT:    addq $32, %rcx
; XOP-NEXT:    cmpq %rcx, %rdx
; XOP-NEXT:    jne .LBB0_4
; XOP-NEXT:  # %bb.5: # %middle.block
; XOP-NEXT:    cmpl %r9d, %edx
; XOP-NEXT:    jne .LBB0_6
; XOP-NEXT:  .LBB0_9: # %for.cond.cleanup
; XOP-NEXT:    vzeroupper
; XOP-NEXT:    retq
; XOP-NEXT:    .p2align 4, 0x90
; XOP-NEXT:  .LBB0_8: # %for.body
; XOP-NEXT:    # in Loop: Header=BB0_6 Depth=1
; XOP-NEXT:    # kill: def $cl killed $cl killed $ecx
; XOP-NEXT:    shll %cl, (%rdi,%rdx,4)
; XOP-NEXT:    incq %rdx
; XOP-NEXT:    cmpq %rdx, %r9
; XOP-NEXT:    je .LBB0_9
; XOP-NEXT:  .LBB0_6: # %for.body
; XOP-NEXT:    # =>This Inner Loop Header: Depth=1
; XOP-NEXT:    cmpb $0, (%rsi,%rdx)
; XOP-NEXT:    movl %eax, %ecx
; XOP-NEXT:    je .LBB0_8
; XOP-NEXT:  # %bb.7: # %for.body
; XOP-NEXT:    # in Loop: Header=BB0_6 Depth=1
; XOP-NEXT:    movl %r8d, %ecx
; XOP-NEXT:    jmp .LBB0_8
entry:
  %cmp12 = icmp sgt i32 %count, 0
  br i1 %cmp12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %count to i64
  %min.iters.check = icmp ult i32 %count, 32
  br i1 %min.iters.check, label %for.body.preheader40, label %vector.ph

for.body.preheader40:
  %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

vector.ph:
  %n.vec = and i64 %wide.trip.count, 4294967264
  %broadcast.splatinsert20 = insertelement <8 x i32> undef, i32 %amt0, i32 0
  %broadcast.splat21 = shufflevector <8 x i32> %broadcast.splatinsert20, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert22 = insertelement <8 x i32> undef, i32 %amt1, i32 0
  %broadcast.splat23 = shufflevector <8 x i32> %broadcast.splatinsert22, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert24 = insertelement <8 x i32> undef, i32 %amt0, i32 0
  %broadcast.splat25 = shufflevector <8 x i32> %broadcast.splatinsert24, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert26 = insertelement <8 x i32> undef, i32 %amt1, i32 0
  %broadcast.splat27 = shufflevector <8 x i32> %broadcast.splatinsert26, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert28 = insertelement <8 x i32> undef, i32 %amt0, i32 0
  %broadcast.splat29 = shufflevector <8 x i32> %broadcast.splatinsert28, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert30 = insertelement <8 x i32> undef, i32 %amt1, i32 0
  %broadcast.splat31 = shufflevector <8 x i32> %broadcast.splatinsert30, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert32 = insertelement <8 x i32> undef, i32 %amt0, i32 0
  %broadcast.splat33 = shufflevector <8 x i32> %broadcast.splatinsert32, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert34 = insertelement <8 x i32> undef, i32 %amt1, i32 0
  %broadcast.splat35 = shufflevector <8 x i32> %broadcast.splatinsert34, <8 x i32> undef, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8, i8* %control, i64 %index
  %1 = bitcast i8* %0 to <8 x i8>*
  %wide.load = load <8 x i8>, <8 x i8>* %1, align 1
  %2 = getelementptr inbounds i8, i8* %0, i64 8
  %3 = bitcast i8* %2 to <8 x i8>*
  %wide.load17 = load <8 x i8>, <8 x i8>* %3, align 1
  %4 = getelementptr inbounds i8, i8* %0, i64 16
  %5 = bitcast i8* %4 to <8 x i8>*
  %wide.load18 = load <8 x i8>, <8 x i8>* %5, align 1
  %6 = getelementptr inbounds i8, i8* %0, i64 24
  %7 = bitcast i8* %6 to <8 x i8>*
  %wide.load19 = load <8 x i8>, <8 x i8>* %7, align 1
  %8 = icmp eq <8 x i8> %wide.load, zeroinitializer
  %9 = icmp eq <8 x i8> %wide.load17, zeroinitializer
  %10 = icmp eq <8 x i8> %wide.load18, zeroinitializer
  %11 = icmp eq <8 x i8> %wide.load19, zeroinitializer
  %12 = select <8 x i1> %8, <8 x i32> %broadcast.splat21, <8 x i32> %broadcast.splat23
  %13 = select <8 x i1> %9, <8 x i32> %broadcast.splat25, <8 x i32> %broadcast.splat27
  %14 = select <8 x i1> %10, <8 x i32> %broadcast.splat29, <8 x i32> %broadcast.splat31
  %15 = select <8 x i1> %11, <8 x i32> %broadcast.splat33, <8 x i32> %broadcast.splat35
  %16 = getelementptr inbounds i32, i32* %arr, i64 %index
  %17 = bitcast i32* %16 to <8 x i32>*
  %wide.load36 = load <8 x i32>, <8 x i32>* %17, align 4
  %18 = getelementptr inbounds i32, i32* %16, i64 8
  %19 = bitcast i32* %18 to <8 x i32>*
  %wide.load37 = load <8 x i32>, <8 x i32>* %19, align 4
  %20 = getelementptr inbounds i32, i32* %16, i64 16
  %21 = bitcast i32* %20 to <8 x i32>*
  %wide.load38 = load <8 x i32>, <8 x i32>* %21, align 4
  %22 = getelementptr inbounds i32, i32* %16, i64 24
  %23 = bitcast i32* %22 to <8 x i32>*
  %wide.load39 = load <8 x i32>, <8 x i32>* %23, align 4
  %24 = shl <8 x i32> %wide.load36, %12
  %25 = shl <8 x i32> %wide.load37, %13
  %26 = shl <8 x i32> %wide.load38, %14
  %27 = shl <8 x i32> %wide.load39, %15
  %28 = bitcast i32* %16 to <8 x i32>*
  store <8 x i32> %24, <8 x i32>* %28, align 4
  %29 = bitcast i32* %18 to <8 x i32>*
  store <8 x i32> %25, <8 x i32>* %29, align 4
  %30 = bitcast i32* %20 to <8 x i32>*
  store <8 x i32> %26, <8 x i32>* %30, align 4
  %31 = bitcast i32* %22 to <8 x i32>*
  store <8 x i32> %27, <8 x i32>* %31, align 4
  %index.next = add i64 %index, 32
  %32 = icmp eq i64 %index.next, %n.vec
  br i1 %32, label %middle.block, label %vector.body

middle.block:
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader40

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader40 ]
  %arrayidx = getelementptr inbounds i8, i8* %control, i64 %indvars.iv
  %33 = load i8, i8* %arrayidx, align 1
  %tobool = icmp eq i8 %33, 0
  %cond = select i1 %tobool, i32 %amt0, i32 %amt1
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %indvars.iv
  %34 = load i32, i32* %arrayidx2, align 4
  %shl = shl i32 %34, %cond
  store i32 %shl, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @vector_variable_shift_left_loop_simpler(i32* nocapture %arr, i8* nocapture readonly %control, i32 %count, i32 %amt0, i32 %amt1, i32 %x) nounwind {
; SSE-LABEL: vector_variable_shift_left_loop_simpler:
; SSE:       # %bb.0: # %entry
; SSE-NEXT:    testl %edx, %edx
; SSE-NEXT:    jle .LBB1_3
; SSE-NEXT:  # %bb.1: # %vector.ph
; SSE-NEXT:    movl %edx, %eax
; SSE-NEXT:    andl $-4, %eax
; SSE-NEXT:    movd %ecx, %xmm0
; SSE-NEXT:    movd %r8d, %xmm3
; SSE-NEXT:    movd %r9d, %xmm1
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,0,0,0]
; SSE-NEXT:    xorl %ecx, %ecx
; SSE-NEXT:    pmovzxdq {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; SSE-NEXT:    movdqa %xmm1, %xmm2
; SSE-NEXT:    pslld %xmm0, %xmm2
; SSE-NEXT:    pmovzxdq {{.*#+}} xmm0 = xmm3[0],zero,xmm3[1],zero
; SSE-NEXT:    pslld %xmm0, %xmm1
; SSE-NEXT:    pxor %xmm3, %xmm3
; SSE-NEXT:    .p2align 4, 0x90
; SSE-NEXT:  .LBB1_2: # %vector.body
; SSE-NEXT:    # =>This Inner Loop Header: Depth=1
; SSE-NEXT:    pmovzxbd {{.*#+}} xmm0 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero
; SSE-NEXT:    pcmpeqd %xmm3, %xmm0
; SSE-NEXT:    movdqa %xmm1, %xmm4
; SSE-NEXT:    blendvps %xmm0, %xmm2, %xmm4
; SSE-NEXT:    movups %xmm4, (%rdi,%rcx,4)
; SSE-NEXT:    addq $4, %rcx
; SSE-NEXT:    cmpq %rcx, %rax
; SSE-NEXT:    jne .LBB1_2
; SSE-NEXT:  .LBB1_3: # %exit
; SSE-NEXT:    retq
;
; AVX1-LABEL: vector_variable_shift_left_loop_simpler:
; AVX1:       # %bb.0: # %entry
; AVX1-NEXT:    testl %edx, %edx
; AVX1-NEXT:    jle .LBB1_3
; AVX1-NEXT:  # %bb.1: # %vector.ph
; AVX1-NEXT:    movl %edx, %eax
; AVX1-NEXT:    andl $-4, %eax
; AVX1-NEXT:    vmovd %ecx, %xmm0
; AVX1-NEXT:    vmovd %r8d, %xmm1
; AVX1-NEXT:    vmovd %r9d, %xmm2
; AVX1-NEXT:    vpshufd {{.*#+}} xmm2 = xmm2[0,0,0,0]
; AVX1-NEXT:    xorl %ecx, %ecx
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; AVX1-NEXT:    vpslld %xmm0, %xmm2, %xmm0
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm1 = xmm1[0],zero,xmm1[1],zero
; AVX1-NEXT:    vpslld %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    .p2align 4, 0x90
; AVX1-NEXT:  .LBB1_2: # %vector.body
; AVX1-NEXT:    # =>This Inner Loop Header: Depth=1
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm3 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero
; AVX1-NEXT:    vpcmpeqd %xmm2, %xmm3, %xmm3
; AVX1-NEXT:    vblendvps %xmm3, %xmm0, %xmm1, %xmm3
; AVX1-NEXT:    vmovups %xmm3, (%rdi,%rcx,4)
; AVX1-NEXT:    addq $4, %rcx
; AVX1-NEXT:    cmpq %rcx, %rax
; AVX1-NEXT:    jne .LBB1_2
; AVX1-NEXT:  .LBB1_3: # %exit
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vector_variable_shift_left_loop_simpler:
; AVX2:       # %bb.0: # %entry
; AVX2-NEXT:    testl %edx, %edx
; AVX2-NEXT:    jle .LBB1_3
; AVX2-NEXT:  # %bb.1: # %vector.ph
; AVX2-NEXT:    movl %edx, %eax
; AVX2-NEXT:    andl $-4, %eax
; AVX2-NEXT:    vmovd %ecx, %xmm0
; AVX2-NEXT:    vpbroadcastd %xmm0, %xmm0
; AVX2-NEXT:    vmovd %r8d, %xmm1
; AVX2-NEXT:    vpbroadcastd %xmm1, %xmm1
; AVX2-NEXT:    vmovd %r9d, %xmm2
; AVX2-NEXT:    vpbroadcastd %xmm2, %xmm2
; AVX2-NEXT:    xorl %ecx, %ecx
; AVX2-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX2-NEXT:    .p2align 4, 0x90
; AVX2-NEXT:  .LBB1_2: # %vector.body
; AVX2-NEXT:    # =>This Inner Loop Header: Depth=1
; AVX2-NEXT:    vpmovzxbd {{.*#+}} xmm4 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero
; AVX2-NEXT:    vpcmpeqd %xmm3, %xmm4, %xmm4
; AVX2-NEXT:    vblendvps %xmm4, %xmm0, %xmm1, %xmm4
; AVX2-NEXT:    vpsllvd %xmm4, %xmm2, %xmm4
; AVX2-NEXT:    vmovdqu %xmm4, (%rdi,%rcx,4)
; AVX2-NEXT:    addq $4, %rcx
; AVX2-NEXT:    cmpq %rcx, %rax
; AVX2-NEXT:    jne .LBB1_2
; AVX2-NEXT:  .LBB1_3: # %exit
; AVX2-NEXT:    retq
;
; XOP-LABEL: vector_variable_shift_left_loop_simpler:
; XOP:       # %bb.0: # %entry
; XOP-NEXT:    testl %edx, %edx
; XOP-NEXT:    jle .LBB1_3
; XOP-NEXT:  # %bb.1: # %vector.ph
; XOP-NEXT:    movl %edx, %eax
; XOP-NEXT:    andl $-4, %eax
; XOP-NEXT:    vmovd %ecx, %xmm0
; XOP-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,0,0,0]
; XOP-NEXT:    vmovd %r8d, %xmm1
; XOP-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,0,0,0]
; XOP-NEXT:    vmovd %r9d, %xmm2
; XOP-NEXT:    vpshufd {{.*#+}} xmm2 = xmm2[0,0,0,0]
; XOP-NEXT:    xorl %ecx, %ecx
; XOP-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; XOP-NEXT:    .p2align 4, 0x90
; XOP-NEXT:  .LBB1_2: # %vector.body
; XOP-NEXT:    # =>This Inner Loop Header: Depth=1
; XOP-NEXT:    vpmovzxbd {{.*#+}} xmm4 = mem[0],zero,zero,zero,mem[1],zero,zero,zero,mem[2],zero,zero,zero,mem[3],zero,zero,zero
; XOP-NEXT:    vpcomeqd %xmm3, %xmm4, %xmm4
; XOP-NEXT:    vblendvps %xmm4, %xmm0, %xmm1, %xmm4
; XOP-NEXT:    vpshld %xmm4, %xmm2, %xmm4
; XOP-NEXT:    vmovdqu %xmm4, (%rdi,%rcx,4)
; XOP-NEXT:    addq $4, %rcx
; XOP-NEXT:    cmpq %rcx, %rax
; XOP-NEXT:    jne .LBB1_2
; XOP-NEXT:  .LBB1_3: # %exit
; XOP-NEXT:    retq
entry:
  %cmp16 = icmp sgt i32 %count, 0
  %wide.trip.count = zext i32 %count to i64
  br i1 %cmp16, label %vector.ph, label %exit

vector.ph:
  %n.vec = and i64 %wide.trip.count, 4294967292
  %splatinsert18 = insertelement <4 x i32> undef, i32 %amt0, i32 0
  %splat1 = shufflevector <4 x i32> %splatinsert18, <4 x i32> undef, <4 x i32> zeroinitializer
  %splatinsert20 = insertelement <4 x i32> undef, i32 %amt1, i32 0
  %splat2 = shufflevector <4 x i32> %splatinsert20, <4 x i32> undef, <4 x i32> zeroinitializer
  %splatinsert22 = insertelement <4 x i32> undef, i32 %x, i32 0
  %splat3 = shufflevector <4 x i32> %splatinsert22, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8, i8* %control, i64 %index
  %1 = bitcast i8* %0 to <4 x i8>*
  %wide.load = load <4 x i8>, <4 x i8>* %1, align 1
  %2 = icmp eq <4 x i8> %wide.load, zeroinitializer
  %3 = select <4 x i1> %2, <4 x i32> %splat1, <4 x i32> %splat2
  %4 = shl <4 x i32> %splat3, %3
  %5 = getelementptr inbounds i32, i32* %arr, i64 %index
  %6 = bitcast i32* %5 to <4 x i32>*
  store <4 x i32> %4, <4 x i32>* %6, align 4
  %index.next = add i64 %index, 4
  %7 = icmp eq i64 %index.next, %n.vec
  br i1 %7, label %exit, label %vector.body

exit:
  ret void
}
