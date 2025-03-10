; RUN: llc -mtriple=i686 %s -o - | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: llc -mtriple=x86_64 %s -o - | FileCheck --check-prefixes=CHECK,X64 %s

;; -fpatchable-function-entry=0 -fcf-protection=branch
define void @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: # %bb.0:
; X86-NEXT:     endbr32
; X64-NEXT:     endbr64
; CHECK-NEXT:  ret
; CHECK-NOT:  .section __patchable_function_entries
  ret void
}

;; -fpatchable-function-entry=1 -fcf-protection=branch
;; For M=0, place the label .Lpatch0 after the initial ENDBR.
;; .cfi_startproc should be placed at the function entry.
define void @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: # %bb.0:
; X86-NEXT:     endbr32
; X64-NEXT:     endbr64
; CHECK-NEXT: .Lpatch0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f1{{$}}
; X86-NEXT:    .p2align 2
; X86-NEXT:    .long .Lpatch0
; X64-NEXT:    .p2align 3
; X64-NEXT:    .quad .Lpatch0
  ret void
}

;; -fpatchable-function-entry=2,1 -fcf-protection=branch
define void @f2_1() "patchable-function-entry"="1" "patchable-function-prefix"="1" {
; CHECK-LABEL: .type f2_1,@function
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT:  nop
; CHECK-NEXT: f2_1:
; CHECK-NEXT: .Lfunc_begin2:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: # %bb.0:
; X86-NEXT:     endbr32
; X64-NEXT:     endbr64
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
; CHECK:      .Lfunc_end2:
; CHECK-NEXT: .size f2_1, .Lfunc_end2-f2_1
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f2_1{{$}}
; X86-NEXT:    .p2align 2
; X86-NEXT:    .long .Ltmp0
; X64-NEXT:    .p2align 3
; X64-NEXT:    .quad .Ltmp0
  ret void
}

;; -fpatchable-function-entry=1 -fcf-protection=branch
;; For M=0, don't create .Lpatch0 if the initial instruction is not ENDBR,
;; even if other basic blocks may have ENDBR.
@buf = internal global [5 x i8*] zeroinitializer
declare i32 @llvm.eh.sjlj.setjmp(i8*)

define internal void @f1i() "patchable-function-entry"="1" {
; CHECK-LABEL: f1i:
; CHECK-NEXT: .Lfunc_begin3:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT:  nop
; CHECK-NOT:  .Lpatch0:
;; Another basic block has ENDBR, but it doesn't affect our decision to not create .Lpatch0
; CHECK:       endbr
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f1i{{$}}
; X86-NEXT:    .p2align 2
; X86-NEXT:    .long .Lfunc_begin3
; X64-NEXT:    .p2align 3
; X64-NEXT:    .quad .Lfunc_begin3
entry:
  tail call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([5 x i8*]* @buf to i8*))
  ret void
}

;; Test the interaction with -fsanitize=function.
; CHECK:      .type sanitize_function,@function
; CHECK-NEXT: .Ltmp{{.*}}:
; CHECK-NEXT:   nop
; CHECK-NEXT:   .long   3238382334
; CHECK-NEXT:   .long   42
; CHECK-NEXT: sanitize_function:
; CHECK-NEXT: .Lfunc_begin{{.*}}:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:   # %bb.0:
; X86-NEXT:      endbr32
; X64-NEXT:      endbr64
; CHECK-NEXT:   nop
; CHECK-NEXT:   ret
define void @sanitize_function(ptr noundef %x) "patchable-function-prefix"="1" "patchable-function-entry"="1" !func_sanitize !1 {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 3238382334, i32 42}
