; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s | FileCheck %s

; LSR should be able to eliminate the max computations by
; making the loops use slt/ult comparisons instead of ne comparisons.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"

define void @foo(i8* %r, i32 %s, i32 %w, i32 %x, i8* %j, i32 %d) nounwind {
; CHECK-LABEL: foo:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:    pushl %esi
; CHECK-NEXT:    subl $28, %esp
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edi
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ebp
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    movl %edi, %ecx
; CHECK-NEXT:    imull %ebp, %ecx
; CHECK-NEXT:    cmpl $1, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl %ecx, (%esp) ## 4-byte Spill
; CHECK-NEXT:    je LBB0_19
; CHECK-NEXT:  ## %bb.1: ## %bb10.preheader
; CHECK-NEXT:    movl %ecx, %eax
; CHECK-NEXT:    sarl $31, %eax
; CHECK-NEXT:    shrl $30, %eax
; CHECK-NEXT:    addl %ecx, %eax
; CHECK-NEXT:    sarl $2, %eax
; CHECK-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    jle LBB0_12
; CHECK-NEXT:  ## %bb.2: ## %bb.nph9
; CHECK-NEXT:    testl %ebp, %ebp
; CHECK-NEXT:    jle LBB0_12
; CHECK-NEXT:  ## %bb.3: ## %bb.nph9.split
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    incl %eax
; CHECK-NEXT:    xorl %ecx, %ecx
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB0_4: ## %bb6
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    movzbl (%eax,%esi,2), %ebx
; CHECK-NEXT:    movb %bl, (%edx,%esi)
; CHECK-NEXT:    incl %esi
; CHECK-NEXT:    cmpl %ebp, %esi
; CHECK-NEXT:    jl LBB0_4
; CHECK-NEXT:  ## %bb.5: ## %bb9
; CHECK-NEXT:    ## in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    incl %ecx
; CHECK-NEXT:    addl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    addl %ebp, %edx
; CHECK-NEXT:    cmpl %edi, %ecx
; CHECK-NEXT:    je LBB0_12
; CHECK-NEXT:  ## %bb.6: ## %bb7.preheader
; CHECK-NEXT:    ## in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    jmp LBB0_4
; CHECK-NEXT:  LBB0_12: ## %bb18.loopexit
; CHECK-NEXT:    movl (%esp), %eax ## 4-byte Reload
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %ecx ## 4-byte Reload
; CHECK-NEXT:    addl %ecx, %eax
; CHECK-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    cmpl $1, %edi
; CHECK-NEXT:    jle LBB0_13
; CHECK-NEXT:  ## %bb.7: ## %bb.nph5
; CHECK-NEXT:    cmpl $2, %ebp
; CHECK-NEXT:    jl LBB0_13
; CHECK-NEXT:  ## %bb.8: ## %bb.nph5.split
; CHECK-NEXT:    movl %ebp, %edx
; CHECK-NEXT:    shrl $31, %edx
; CHECK-NEXT:    addl %ebp, %edx
; CHECK-NEXT:    sarl %edx
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $31, %ecx
; CHECK-NEXT:    addl %eax, %ecx
; CHECK-NEXT:    sarl %ecx
; CHECK-NEXT:    movl %ecx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %eax ## 4-byte Reload
; CHECK-NEXT:    addl %ecx, %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %esi
; CHECK-NEXT:    addl $2, %esi
; CHECK-NEXT:    movl %esi, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    movl (%esp), %esi ## 4-byte Reload
; CHECK-NEXT:    addl %esi, %ecx
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    xorl %edi, %edi
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB0_9: ## %bb13
; CHECK-NEXT:    ## =>This Loop Header: Depth=1
; CHECK-NEXT:    ## Child Loop BB0_10 Depth 2
; CHECK-NEXT:    movl %edi, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    andl $1, %edi
; CHECK-NEXT:    movl %esi, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    addl %esi, %edi
; CHECK-NEXT:    imull {{[0-9]+}}(%esp), %edi
; CHECK-NEXT:    addl {{[-0-9]+}}(%e{{[sb]}}p), %edi ## 4-byte Folded Reload
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB0_10: ## %bb14
; CHECK-NEXT:    ## Parent Loop BB0_9 Depth=1
; CHECK-NEXT:    ## => This Inner Loop Header: Depth=2
; CHECK-NEXT:    movzbl -2(%edi,%esi,4), %ebx
; CHECK-NEXT:    movb %bl, (%ecx,%esi)
; CHECK-NEXT:    movzbl (%edi,%esi,4), %ebx
; CHECK-NEXT:    movb %bl, (%eax,%esi)
; CHECK-NEXT:    incl %esi
; CHECK-NEXT:    cmpl %edx, %esi
; CHECK-NEXT:    jl LBB0_10
; CHECK-NEXT:  ## %bb.11: ## %bb17
; CHECK-NEXT:    ## in Loop: Header=BB0_9 Depth=1
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %edi ## 4-byte Reload
; CHECK-NEXT:    incl %edi
; CHECK-NEXT:    addl %edx, %eax
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %esi ## 4-byte Reload
; CHECK-NEXT:    addl $2, %esi
; CHECK-NEXT:    addl %edx, %ecx
; CHECK-NEXT:    cmpl {{[-0-9]+}}(%e{{[sb]}}p), %edi ## 4-byte Folded Reload
; CHECK-NEXT:    jl LBB0_9
; CHECK-NEXT:  LBB0_13: ## %bb20
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    cmpl $1, %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edi
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    je LBB0_19
; CHECK-NEXT:  ## %bb.14: ## %bb20
; CHECK-NEXT:    cmpl $3, %eax
; CHECK-NEXT:    jne LBB0_24
; CHECK-NEXT:  ## %bb.15: ## %bb22
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %ebx ## 4-byte Reload
; CHECK-NEXT:    addl %ebx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Folded Spill
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    jle LBB0_18
; CHECK-NEXT:  ## %bb.16: ## %bb.nph
; CHECK-NEXT:    leal 15(%edi), %eax
; CHECK-NEXT:    andl $-16, %eax
; CHECK-NEXT:    imull {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    addl %ebx, %ebx
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movl (%esp), %esi ## 4-byte Reload
; CHECK-NEXT:    addl %esi, %ecx
; CHECK-NEXT:    addl %ecx, %ebx
; CHECK-NEXT:    addl %eax, %edx
; CHECK-NEXT:    leal 15(%ebp), %eax
; CHECK-NEXT:    andl $-16, %eax
; CHECK-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB0_17: ## %bb23
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    pushl %edx
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:    movl %ebx, %esi
; CHECK-NEXT:    movl %edx, %ebx
; CHECK-NEXT:    calll _memcpy
; CHECK-NEXT:    movl %ebx, %edx
; CHECK-NEXT:    movl %esi, %ebx
; CHECK-NEXT:    addl $16, %esp
; CHECK-NEXT:    addl %ebp, %ebx
; CHECK-NEXT:    addl {{[-0-9]+}}(%e{{[sb]}}p), %edx ## 4-byte Folded Reload
; CHECK-NEXT:    decl %edi
; CHECK-NEXT:    jne LBB0_17
; CHECK-NEXT:  LBB0_18: ## %bb26
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %eax ## 4-byte Reload
; CHECK-NEXT:    movl (%esp), %edx ## 4-byte Reload
; CHECK-NEXT:    addl %edx, %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    addl %eax, %ecx
; CHECK-NEXT:    jmp LBB0_23
; CHECK-NEXT:  LBB0_19: ## %bb29
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    jle LBB0_22
; CHECK-NEXT:  ## %bb.20: ## %bb.nph11
; CHECK-NEXT:    movl %edi, %esi
; CHECK-NEXT:    leal 15(%ebp), %eax
; CHECK-NEXT:    andl $-16, %eax
; CHECK-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edi
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB0_21: ## %bb30
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    pushl %edx
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:    movl %edx, %ebx
; CHECK-NEXT:    calll _memcpy
; CHECK-NEXT:    movl %ebx, %edx
; CHECK-NEXT:    addl $16, %esp
; CHECK-NEXT:    addl %ebp, %edi
; CHECK-NEXT:    addl {{[-0-9]+}}(%e{{[sb]}}p), %edx ## 4-byte Folded Reload
; CHECK-NEXT:    decl %esi
; CHECK-NEXT:    jne LBB0_21
; CHECK-NEXT:  LBB0_22: ## %bb33
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movl (%esp), %edx ## 4-byte Reload
; CHECK-NEXT:    addl %edx, %ecx
; CHECK-NEXT:  LBB0_23: ## %bb33
; CHECK-NEXT:    movl %edx, %eax
; CHECK-NEXT:    shrl $31, %eax
; CHECK-NEXT:    addl %edx, %eax
; CHECK-NEXT:    sarl %eax
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    pushl $128
; CHECK-NEXT:    pushl %ecx
; CHECK-NEXT:    calll _memset
; CHECK-NEXT:    addl $44, %esp
; CHECK-NEXT:  LBB0_25: ## %return
; CHECK-NEXT:    popl %esi
; CHECK-NEXT:    popl %edi
; CHECK-NEXT:    popl %ebx
; CHECK-NEXT:    popl %ebp
; CHECK-NEXT:    retl
; CHECK-NEXT:  LBB0_24: ## %return
; CHECK-NEXT:    addl $28, %esp
; CHECK-NEXT:    jmp LBB0_25
entry:
  %0 = mul i32 %x, %w
  %1 = mul i32 %x, %w
  %2 = sdiv i32 %1, 4
  %.sum2 = add i32 %2, %0
  %cond = icmp eq i32 %d, 1
  br i1 %cond, label %bb29, label %bb10.preheader

bb10.preheader:                                   ; preds = %entry
  %3 = icmp sgt i32 %x, 0
  br i1 %3, label %bb.nph9, label %bb18.loopexit

bb.nph7:                                          ; preds = %bb7.preheader
  %4 = mul i32 %y.08, %w
  %5 = mul i32 %y.08, %s
  %6 = add i32 %5, 1
  %tmp8 = icmp sgt i32 1, %w
  %smax9 = select i1 %tmp8, i32 1, i32 %w
  br label %bb6

bb6:                                              ; preds = %bb7, %bb.nph7
  %x.06 = phi i32 [ 0, %bb.nph7 ], [ %indvar.next7, %bb7 ]
  %7 = add i32 %x.06, %4
  %8 = shl i32 %x.06, 1
  %9 = add i32 %6, %8
  %10 = getelementptr i8, i8* %r, i32 %9
  %11 = load i8, i8* %10, align 1
  %12 = getelementptr i8, i8* %j, i32 %7
  store i8 %11, i8* %12, align 1
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next7 = add i32 %x.06, 1
  %exitcond10 = icmp ne i32 %indvar.next7, %smax9
  br i1 %exitcond10, label %bb6, label %bb7.bb9_crit_edge

bb7.bb9_crit_edge:                                ; preds = %bb7
  br label %bb9

bb9:                                              ; preds = %bb7.preheader, %bb7.bb9_crit_edge
  br label %bb10

bb10:                                             ; preds = %bb9
  %indvar.next11 = add i32 %y.08, 1
  %exitcond12 = icmp ne i32 %indvar.next11, %x
  br i1 %exitcond12, label %bb7.preheader, label %bb10.bb18.loopexit_crit_edge

bb10.bb18.loopexit_crit_edge:                     ; preds = %bb10
  br label %bb10.bb18.loopexit_crit_edge.split

bb10.bb18.loopexit_crit_edge.split:               ; preds = %bb.nph9, %bb10.bb18.loopexit_crit_edge
  br label %bb18.loopexit

bb.nph9:                                          ; preds = %bb10.preheader
  %13 = icmp sgt i32 %w, 0
  br i1 %13, label %bb.nph9.split, label %bb10.bb18.loopexit_crit_edge.split

bb.nph9.split:                                    ; preds = %bb.nph9
  br label %bb7.preheader

bb7.preheader:                                    ; preds = %bb.nph9.split, %bb10
  %y.08 = phi i32 [ 0, %bb.nph9.split ], [ %indvar.next11, %bb10 ]
  br i1 true, label %bb.nph7, label %bb9

bb.nph5:                                          ; preds = %bb18.loopexit
  %14 = sdiv i32 %w, 2
  %15 = icmp slt i32 %w, 2
  %16 = sdiv i32 %x, 2
  br i1 %15, label %bb18.bb20_crit_edge.split, label %bb.nph5.split

bb.nph5.split:                                    ; preds = %bb.nph5
  %tmp2 = icmp sgt i32 1, %16
  %smax3 = select i1 %tmp2, i32 1, i32 %16
  br label %bb13

bb13:                                             ; preds = %bb18, %bb.nph5.split
  %y.14 = phi i32 [ 0, %bb.nph5.split ], [ %indvar.next1, %bb18 ]
  %17 = mul i32 %14, %y.14
  %18 = shl i32 %y.14, 1
  %19 = srem i32 %y.14, 2
  %20 = add i32 %19, %18
  %21 = mul i32 %20, %s
  br i1 true, label %bb.nph3, label %bb17

bb.nph3:                                          ; preds = %bb13
  %22 = add i32 %17, %0
  %23 = add i32 %17, %.sum2
  %24 = sdiv i32 %w, 2
  %tmp = icmp sgt i32 1, %24
  %smax = select i1 %tmp, i32 1, i32 %24
  br label %bb14

bb14:                                             ; preds = %bb15, %bb.nph3
  %x.12 = phi i32 [ 0, %bb.nph3 ], [ %indvar.next, %bb15 ]
  %25 = shl i32 %x.12, 2
  %26 = add i32 %25, %21
  %27 = getelementptr i8, i8* %r, i32 %26
  %28 = load i8, i8* %27, align 1
  %.sum = add i32 %22, %x.12
  %29 = getelementptr i8, i8* %j, i32 %.sum
  store i8 %28, i8* %29, align 1
  %30 = shl i32 %x.12, 2
  %31 = or disjoint i32 %30, 2
  %32 = add i32 %31, %21
  %33 = getelementptr i8, i8* %r, i32 %32
  %34 = load i8, i8* %33, align 1
  %.sum6 = add i32 %23, %x.12
  %35 = getelementptr i8, i8* %j, i32 %.sum6
  store i8 %34, i8* %35, align 1
  br label %bb15

bb15:                                             ; preds = %bb14
  %indvar.next = add i32 %x.12, 1
  %exitcond = icmp ne i32 %indvar.next, %smax
  br i1 %exitcond, label %bb14, label %bb15.bb17_crit_edge

bb15.bb17_crit_edge:                              ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb15.bb17_crit_edge, %bb13
  br label %bb18

bb18.loopexit:                                    ; preds = %bb10.bb18.loopexit_crit_edge.split, %bb10.preheader
  %36 = icmp slt i32 %x, 2
  br i1 %36, label %bb20, label %bb.nph5

bb18:                                             ; preds = %bb17
  %indvar.next1 = add i32 %y.14, 1
  %exitcond4 = icmp ne i32 %indvar.next1, %smax3
  br i1 %exitcond4, label %bb13, label %bb18.bb20_crit_edge

bb18.bb20_crit_edge:                              ; preds = %bb18
  br label %bb18.bb20_crit_edge.split

bb18.bb20_crit_edge.split:                        ; preds = %bb18.bb20_crit_edge, %bb.nph5
  br label %bb20

bb20:                                             ; preds = %bb18.bb20_crit_edge.split, %bb18.loopexit
  switch i32 %d, label %return [
    i32 3, label %bb22
    i32 1, label %bb29
  ]

bb22:                                             ; preds = %bb20
  %37 = mul i32 %x, %w
  %38 = sdiv i32 %37, 4
  %.sum3 = add i32 %38, %.sum2
  %39 = add i32 %x, 15
  %40 = and i32 %39, -16
  %41 = add i32 %w, 15
  %42 = and i32 %41, -16
  %43 = mul i32 %40, %s
  %44 = icmp sgt i32 %x, 0
  br i1 %44, label %bb.nph, label %bb26

bb.nph:                                           ; preds = %bb22
  br label %bb23

bb23:                                             ; preds = %bb24, %bb.nph
  %y.21 = phi i32 [ 0, %bb.nph ], [ %indvar.next5, %bb24 ]
  %45 = mul i32 %y.21, %42
  %.sum1 = add i32 %45, %43
  %46 = getelementptr i8, i8* %r, i32 %.sum1
  %47 = mul i32 %y.21, %w
  %.sum5 = add i32 %47, %.sum3
  %48 = getelementptr i8, i8* %j, i32 %.sum5
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %48, i8* %46, i32 %w, i1 false)
  br label %bb24

bb24:                                             ; preds = %bb23
  %indvar.next5 = add i32 %y.21, 1
  %exitcond6 = icmp ne i32 %indvar.next5, %x
  br i1 %exitcond6, label %bb23, label %bb24.bb26_crit_edge

bb24.bb26_crit_edge:                              ; preds = %bb24
  br label %bb26

bb26:                                             ; preds = %bb24.bb26_crit_edge, %bb22
  %49 = mul i32 %x, %w
  %.sum4 = add i32 %.sum3, %49
  %50 = getelementptr i8, i8* %j, i32 %.sum4
  %51 = mul i32 %x, %w
  %52 = sdiv i32 %51, 2
  tail call void @llvm.memset.p0i8.i32(i8* %50, i8 -128, i32 %52, i1 false)
  ret void

bb29:                                             ; preds = %bb20, %entry
  %53 = add i32 %w, 15
  %54 = and i32 %53, -16
  %55 = icmp sgt i32 %x, 0
  br i1 %55, label %bb.nph11, label %bb33

bb.nph11:                                         ; preds = %bb29
  br label %bb30

bb30:                                             ; preds = %bb31, %bb.nph11
  %y.310 = phi i32 [ 0, %bb.nph11 ], [ %indvar.next13, %bb31 ]
  %56 = mul i32 %y.310, %54
  %57 = getelementptr i8, i8* %r, i32 %56
  %58 = mul i32 %y.310, %w
  %59 = getelementptr i8, i8* %j, i32 %58
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %59, i8* %57, i32 %w, i1 false)
  br label %bb31

bb31:                                             ; preds = %bb30
  %indvar.next13 = add i32 %y.310, 1
  %exitcond14 = icmp ne i32 %indvar.next13, %x
  br i1 %exitcond14, label %bb30, label %bb31.bb33_crit_edge

bb31.bb33_crit_edge:                              ; preds = %bb31
  br label %bb33

bb33:                                             ; preds = %bb31.bb33_crit_edge, %bb29
  %60 = mul i32 %x, %w
  %61 = getelementptr i8, i8* %j, i32 %60
  %62 = mul i32 %x, %w
  %63 = sdiv i32 %62, 2
  tail call void @llvm.memset.p0i8.i32(i8* %61, i8 -128, i32 %63, i1 false)
  ret void

return:                                           ; preds = %bb20
  ret void
}

define void @bar(i8* %r, i32 %s, i32 %w, i32 %x, i8* %j, i32 %d) nounwind {
; CHECK-LABEL: bar:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:    pushl %esi
; CHECK-NEXT:    subl $28, %esp
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %esi
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ebp
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movl %ebp, %edx
; CHECK-NEXT:    imull %eax, %edx
; CHECK-NEXT:    cmpl $1, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl %edx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    je LBB1_19
; CHECK-NEXT:  ## %bb.1: ## %bb10.preheader
; CHECK-NEXT:    movl %edx, %ecx
; CHECK-NEXT:    shrl $2, %ecx
; CHECK-NEXT:    movl %ecx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    testl %ebp, %ebp
; CHECK-NEXT:    movl %eax, %edi
; CHECK-NEXT:    je LBB1_12
; CHECK-NEXT:  ## %bb.2: ## %bb.nph9
; CHECK-NEXT:    testl %eax, %eax
; CHECK-NEXT:    je LBB1_12
; CHECK-NEXT:  ## %bb.3: ## %bb.nph9.split
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    incl %eax
; CHECK-NEXT:    xorl %ecx, %ecx
; CHECK-NEXT:    movl %esi, %edx
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB1_4: ## %bb6
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    movzbl (%eax,%esi,2), %ebx
; CHECK-NEXT:    movb %bl, (%edx,%esi)
; CHECK-NEXT:    incl %esi
; CHECK-NEXT:    cmpl %edi, %esi
; CHECK-NEXT:    jb LBB1_4
; CHECK-NEXT:  ## %bb.5: ## %bb9
; CHECK-NEXT:    ## in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    incl %ecx
; CHECK-NEXT:    addl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    addl %edi, %edx
; CHECK-NEXT:    cmpl %ebp, %ecx
; CHECK-NEXT:    je LBB1_12
; CHECK-NEXT:  ## %bb.6: ## %bb7.preheader
; CHECK-NEXT:    ## in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    jmp LBB1_4
; CHECK-NEXT:  LBB1_12: ## %bb18.loopexit
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %eax ## 4-byte Reload
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %ecx ## 4-byte Reload
; CHECK-NEXT:    addl %ecx, %eax
; CHECK-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    cmpl $1, %ebp
; CHECK-NEXT:    jbe LBB1_13
; CHECK-NEXT:  ## %bb.7: ## %bb.nph5
; CHECK-NEXT:    cmpl $2, %edi
; CHECK-NEXT:    jb LBB1_13
; CHECK-NEXT:  ## %bb.8: ## %bb.nph5.split
; CHECK-NEXT:    movl %edi, %ebp
; CHECK-NEXT:    shrl %ebp
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    shrl %eax
; CHECK-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %ecx ## 4-byte Reload
; CHECK-NEXT:    addl %eax, %ecx
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    addl $2, %edx
; CHECK-NEXT:    movl %edx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %edx ## 4-byte Reload
; CHECK-NEXT:    addl %edx, %eax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    xorl %ebx, %ebx
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB1_9: ## %bb13
; CHECK-NEXT:    ## =>This Loop Header: Depth=1
; CHECK-NEXT:    ## Child Loop BB1_10 Depth 2
; CHECK-NEXT:    movl %ebx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    andl $1, %ebx
; CHECK-NEXT:    movl %edx, (%esp) ## 4-byte Spill
; CHECK-NEXT:    addl %edx, %ebx
; CHECK-NEXT:    imull {{[0-9]+}}(%esp), %ebx
; CHECK-NEXT:    addl {{[-0-9]+}}(%e{{[sb]}}p), %ebx ## 4-byte Folded Reload
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB1_10: ## %bb14
; CHECK-NEXT:    ## Parent Loop BB1_9 Depth=1
; CHECK-NEXT:    ## => This Inner Loop Header: Depth=2
; CHECK-NEXT:    movzbl -2(%ebx,%esi,4), %edx
; CHECK-NEXT:    movb %dl, (%eax,%esi)
; CHECK-NEXT:    movzbl (%ebx,%esi,4), %edx
; CHECK-NEXT:    movb %dl, (%ecx,%esi)
; CHECK-NEXT:    incl %esi
; CHECK-NEXT:    cmpl %ebp, %esi
; CHECK-NEXT:    jb LBB1_10
; CHECK-NEXT:  ## %bb.11: ## %bb17
; CHECK-NEXT:    ## in Loop: Header=BB1_9 Depth=1
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %ebx ## 4-byte Reload
; CHECK-NEXT:    incl %ebx
; CHECK-NEXT:    addl %ebp, %ecx
; CHECK-NEXT:    movl (%esp), %edx ## 4-byte Reload
; CHECK-NEXT:    addl $2, %edx
; CHECK-NEXT:    addl %ebp, %eax
; CHECK-NEXT:    cmpl {{[-0-9]+}}(%e{{[sb]}}p), %ebx ## 4-byte Folded Reload
; CHECK-NEXT:    jb LBB1_9
; CHECK-NEXT:  LBB1_13: ## %bb20
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %esi
; CHECK-NEXT:    cmpl $1, %esi
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ebp
; CHECK-NEXT:    movl %edi, %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    je LBB1_19
; CHECK-NEXT:  ## %bb.14: ## %bb20
; CHECK-NEXT:    cmpl $3, %esi
; CHECK-NEXT:    jne LBB1_24
; CHECK-NEXT:  ## %bb.15: ## %bb22
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %edx ## 4-byte Reload
; CHECK-NEXT:    addl %edx, {{[-0-9]+}}(%e{{[sb]}}p) ## 4-byte Folded Spill
; CHECK-NEXT:    testl %ebp, %ebp
; CHECK-NEXT:    je LBB1_18
; CHECK-NEXT:  ## %bb.16: ## %bb.nph
; CHECK-NEXT:    movl %ebp, %esi
; CHECK-NEXT:    leal 15(%ebp), %eax
; CHECK-NEXT:    andl $-16, %eax
; CHECK-NEXT:    imull {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    addl $15, %edx
; CHECK-NEXT:    andl $-16, %edx
; CHECK-NEXT:    movl %edx, (%esp) ## 4-byte Spill
; CHECK-NEXT:    addl %eax, %ecx
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %eax ## 4-byte Reload
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    leal (%edx,%eax), %ebp
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB1_17: ## %bb23
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ebx
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:    pushl %ecx
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    movl %ecx, %edi
; CHECK-NEXT:    calll _memcpy
; CHECK-NEXT:    movl %edi, %ecx
; CHECK-NEXT:    addl $16, %esp
; CHECK-NEXT:    addl %ebx, %ebp
; CHECK-NEXT:    addl (%esp), %ecx ## 4-byte Folded Reload
; CHECK-NEXT:    decl %esi
; CHECK-NEXT:    jne LBB1_17
; CHECK-NEXT:  LBB1_18: ## %bb26
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %eax ## 4-byte Reload
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %ecx ## 4-byte Reload
; CHECK-NEXT:    addl %ecx, %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    addl %eax, %edx
; CHECK-NEXT:    shrl %ecx
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    pushl %ecx
; CHECK-NEXT:    pushl $128
; CHECK-NEXT:    pushl %edx
; CHECK-NEXT:    jmp LBB1_23
; CHECK-NEXT:  LBB1_19: ## %bb29
; CHECK-NEXT:    testl %ebp, %ebp
; CHECK-NEXT:    je LBB1_22
; CHECK-NEXT:  ## %bb.20: ## %bb.nph11
; CHECK-NEXT:    movl %ebp, %esi
; CHECK-NEXT:    movl %eax, %edi
; CHECK-NEXT:    addl $15, %eax
; CHECK-NEXT:    andl $-16, %eax
; CHECK-NEXT:    movl %eax, (%esp) ## 4-byte Spill
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ebp
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  LBB1_21: ## %bb30
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:    pushl %ecx
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    movl %ecx, %ebx
; CHECK-NEXT:    calll _memcpy
; CHECK-NEXT:    movl %ebx, %ecx
; CHECK-NEXT:    addl $16, %esp
; CHECK-NEXT:    addl %edi, %ebp
; CHECK-NEXT:    addl (%esp), %ecx ## 4-byte Folded Reload
; CHECK-NEXT:    decl %esi
; CHECK-NEXT:    jne LBB1_21
; CHECK-NEXT:  LBB1_22: ## %bb33
; CHECK-NEXT:    movl {{[-0-9]+}}(%e{{[sb]}}p), %eax ## 4-byte Reload
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    addl %eax, %ecx
; CHECK-NEXT:    shrl %eax
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    pushl $128
; CHECK-NEXT:    pushl %ecx
; CHECK-NEXT:  LBB1_23: ## %bb33
; CHECK-NEXT:    calll _memset
; CHECK-NEXT:    addl $44, %esp
; CHECK-NEXT:  LBB1_25: ## %return
; CHECK-NEXT:    popl %esi
; CHECK-NEXT:    popl %edi
; CHECK-NEXT:    popl %ebx
; CHECK-NEXT:    popl %ebp
; CHECK-NEXT:    retl
; CHECK-NEXT:  LBB1_24: ## %return
; CHECK-NEXT:    addl $28, %esp
; CHECK-NEXT:    jmp LBB1_25
entry:
  %0 = mul i32 %x, %w
  %1 = mul i32 %x, %w
  %2 = udiv i32 %1, 4
  %.sum2 = add i32 %2, %0
  %cond = icmp eq i32 %d, 1
  br i1 %cond, label %bb29, label %bb10.preheader

bb10.preheader:                                   ; preds = %entry
  %3 = icmp ne i32 %x, 0
  br i1 %3, label %bb.nph9, label %bb18.loopexit

bb.nph7:                                          ; preds = %bb7.preheader
  %4 = mul i32 %y.08, %w
  %5 = mul i32 %y.08, %s
  %6 = add i32 %5, 1
  %tmp8 = icmp ugt i32 1, %w
  %smax9 = select i1 %tmp8, i32 1, i32 %w
  br label %bb6

bb6:                                              ; preds = %bb7, %bb.nph7
  %x.06 = phi i32 [ 0, %bb.nph7 ], [ %indvar.next7, %bb7 ]
  %7 = add i32 %x.06, %4
  %8 = shl i32 %x.06, 1
  %9 = add i32 %6, %8
  %10 = getelementptr i8, i8* %r, i32 %9
  %11 = load i8, i8* %10, align 1
  %12 = getelementptr i8, i8* %j, i32 %7
  store i8 %11, i8* %12, align 1
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next7 = add i32 %x.06, 1
  %exitcond10 = icmp ne i32 %indvar.next7, %smax9
  br i1 %exitcond10, label %bb6, label %bb7.bb9_crit_edge

bb7.bb9_crit_edge:                                ; preds = %bb7
  br label %bb9

bb9:                                              ; preds = %bb7.preheader, %bb7.bb9_crit_edge
  br label %bb10

bb10:                                             ; preds = %bb9
  %indvar.next11 = add i32 %y.08, 1
  %exitcond12 = icmp ne i32 %indvar.next11, %x
  br i1 %exitcond12, label %bb7.preheader, label %bb10.bb18.loopexit_crit_edge

bb10.bb18.loopexit_crit_edge:                     ; preds = %bb10
  br label %bb10.bb18.loopexit_crit_edge.split

bb10.bb18.loopexit_crit_edge.split:               ; preds = %bb.nph9, %bb10.bb18.loopexit_crit_edge
  br label %bb18.loopexit

bb.nph9:                                          ; preds = %bb10.preheader
  %13 = icmp ugt i32 %w, 0
  br i1 %13, label %bb.nph9.split, label %bb10.bb18.loopexit_crit_edge.split

bb.nph9.split:                                    ; preds = %bb.nph9
  br label %bb7.preheader

bb7.preheader:                                    ; preds = %bb.nph9.split, %bb10
  %y.08 = phi i32 [ 0, %bb.nph9.split ], [ %indvar.next11, %bb10 ]
  br i1 true, label %bb.nph7, label %bb9

bb.nph5:                                          ; preds = %bb18.loopexit
  %14 = udiv i32 %w, 2
  %15 = icmp ult i32 %w, 2
  %16 = udiv i32 %x, 2
  br i1 %15, label %bb18.bb20_crit_edge.split, label %bb.nph5.split

bb.nph5.split:                                    ; preds = %bb.nph5
  %tmp2 = icmp ugt i32 1, %16
  %smax3 = select i1 %tmp2, i32 1, i32 %16
  br label %bb13

bb13:                                             ; preds = %bb18, %bb.nph5.split
  %y.14 = phi i32 [ 0, %bb.nph5.split ], [ %indvar.next1, %bb18 ]
  %17 = mul i32 %14, %y.14
  %18 = shl i32 %y.14, 1
  %19 = urem i32 %y.14, 2
  %20 = add i32 %19, %18
  %21 = mul i32 %20, %s
  br i1 true, label %bb.nph3, label %bb17

bb.nph3:                                          ; preds = %bb13
  %22 = add i32 %17, %0
  %23 = add i32 %17, %.sum2
  %24 = udiv i32 %w, 2
  %tmp = icmp ugt i32 1, %24
  %smax = select i1 %tmp, i32 1, i32 %24
  br label %bb14

bb14:                                             ; preds = %bb15, %bb.nph3
  %x.12 = phi i32 [ 0, %bb.nph3 ], [ %indvar.next, %bb15 ]
  %25 = shl i32 %x.12, 2
  %26 = add i32 %25, %21
  %27 = getelementptr i8, i8* %r, i32 %26
  %28 = load i8, i8* %27, align 1
  %.sum = add i32 %22, %x.12
  %29 = getelementptr i8, i8* %j, i32 %.sum
  store i8 %28, i8* %29, align 1
  %30 = shl i32 %x.12, 2
  %31 = or disjoint i32 %30, 2
  %32 = add i32 %31, %21
  %33 = getelementptr i8, i8* %r, i32 %32
  %34 = load i8, i8* %33, align 1
  %.sum6 = add i32 %23, %x.12
  %35 = getelementptr i8, i8* %j, i32 %.sum6
  store i8 %34, i8* %35, align 1
  br label %bb15

bb15:                                             ; preds = %bb14
  %indvar.next = add i32 %x.12, 1
  %exitcond = icmp ne i32 %indvar.next, %smax
  br i1 %exitcond, label %bb14, label %bb15.bb17_crit_edge

bb15.bb17_crit_edge:                              ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb15.bb17_crit_edge, %bb13
  br label %bb18

bb18.loopexit:                                    ; preds = %bb10.bb18.loopexit_crit_edge.split, %bb10.preheader
  %36 = icmp ult i32 %x, 2
  br i1 %36, label %bb20, label %bb.nph5

bb18:                                             ; preds = %bb17
  %indvar.next1 = add i32 %y.14, 1
  %exitcond4 = icmp ne i32 %indvar.next1, %smax3
  br i1 %exitcond4, label %bb13, label %bb18.bb20_crit_edge

bb18.bb20_crit_edge:                              ; preds = %bb18
  br label %bb18.bb20_crit_edge.split

bb18.bb20_crit_edge.split:                        ; preds = %bb18.bb20_crit_edge, %bb.nph5
  br label %bb20

bb20:                                             ; preds = %bb18.bb20_crit_edge.split, %bb18.loopexit
  switch i32 %d, label %return [
    i32 3, label %bb22
    i32 1, label %bb29
  ]

bb22:                                             ; preds = %bb20
  %37 = mul i32 %x, %w
  %38 = udiv i32 %37, 4
  %.sum3 = add i32 %38, %.sum2
  %39 = add i32 %x, 15
  %40 = and i32 %39, -16
  %41 = add i32 %w, 15
  %42 = and i32 %41, -16
  %43 = mul i32 %40, %s
  %44 = icmp ugt i32 %x, 0
  br i1 %44, label %bb.nph, label %bb26

bb.nph:                                           ; preds = %bb22
  br label %bb23

bb23:                                             ; preds = %bb24, %bb.nph
  %y.21 = phi i32 [ 0, %bb.nph ], [ %indvar.next5, %bb24 ]
  %45 = mul i32 %y.21, %42
  %.sum1 = add i32 %45, %43
  %46 = getelementptr i8, i8* %r, i32 %.sum1
  %47 = mul i32 %y.21, %w
  %.sum5 = add i32 %47, %.sum3
  %48 = getelementptr i8, i8* %j, i32 %.sum5
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %48, i8* %46, i32 %w, i1 false)
  br label %bb24

bb24:                                             ; preds = %bb23
  %indvar.next5 = add i32 %y.21, 1
  %exitcond6 = icmp ne i32 %indvar.next5, %x
  br i1 %exitcond6, label %bb23, label %bb24.bb26_crit_edge

bb24.bb26_crit_edge:                              ; preds = %bb24
  br label %bb26

bb26:                                             ; preds = %bb24.bb26_crit_edge, %bb22
  %49 = mul i32 %x, %w
  %.sum4 = add i32 %.sum3, %49
  %50 = getelementptr i8, i8* %j, i32 %.sum4
  %51 = mul i32 %x, %w
  %52 = udiv i32 %51, 2
  tail call void @llvm.memset.p0i8.i32(i8* %50, i8 -128, i32 %52, i1 false)
  ret void

bb29:                                             ; preds = %bb20, %entry
  %53 = add i32 %w, 15
  %54 = and i32 %53, -16
  %55 = icmp ugt i32 %x, 0
  br i1 %55, label %bb.nph11, label %bb33

bb.nph11:                                         ; preds = %bb29
  br label %bb30

bb30:                                             ; preds = %bb31, %bb.nph11
  %y.310 = phi i32 [ 0, %bb.nph11 ], [ %indvar.next13, %bb31 ]
  %56 = mul i32 %y.310, %54
  %57 = getelementptr i8, i8* %r, i32 %56
  %58 = mul i32 %y.310, %w
  %59 = getelementptr i8, i8* %j, i32 %58
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %59, i8* %57, i32 %w, i1 false)
  br label %bb31

bb31:                                             ; preds = %bb30
  %indvar.next13 = add i32 %y.310, 1
  %exitcond14 = icmp ne i32 %indvar.next13, %x
  br i1 %exitcond14, label %bb30, label %bb31.bb33_crit_edge

bb31.bb33_crit_edge:                              ; preds = %bb31
  br label %bb33

bb33:                                             ; preds = %bb31.bb33_crit_edge, %bb29
  %60 = mul i32 %x, %w
  %61 = getelementptr i8, i8* %j, i32 %60
  %62 = mul i32 %x, %w
  %63 = udiv i32 %62, 2
  tail call void @llvm.memset.p0i8.i32(i8* %61, i8 -128, i32 %63, i1 false)
  ret void

return:                                           ; preds = %bb20
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
