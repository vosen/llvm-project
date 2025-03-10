; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 4
; RUN: llc -mcpu=corei7 < %s -verify-machineinstrs | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i386-pc-linux"

; Check that eh_return & unwind_init were properly lowered

define ptr @test1(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: test1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    movl %esp, %ebp
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:    pushl %esi
; CHECK-NEXT:    pushl %edx
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    movl 12(%ebp), %ecx
; CHECK-NEXT:    movl 8(%ebp), %eax
; CHECK-NEXT:    movl %ecx, 4(%ebp,%eax)
; CHECK-NEXT:    leal 4(%ebp,%eax), %ecx
; CHECK-NEXT:    addl $4, %esp
; CHECK-NEXT:    popl %eax
; CHECK-NEXT:    popl %edx
; CHECK-NEXT:    popl %esi
; CHECK-NEXT:    popl %edi
; CHECK-NEXT:    popl %ebx
; CHECK-NEXT:    popl %ebp
; CHECK-NEXT:    movl %ecx, %esp
; CHECK-NEXT:    retl # eh_return, addr: %ecx
entry:
  call void @llvm.eh.unwind.init()
  %foo   = alloca i32
  call void @llvm.eh.return.i32(i32 %a, ptr %b)
  unreachable
}

define ptr @test2(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: test2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    movl %esp, %ebp
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    movl 12(%ebp), %ecx
; CHECK-NEXT:    movl 8(%ebp), %eax
; CHECK-NEXT:    movl %ecx, 4(%ebp,%eax)
; CHECK-NEXT:    leal 4(%ebp,%eax), %ecx
; CHECK-NEXT:    popl %eax
; CHECK-NEXT:    popl %ebp
; CHECK-NEXT:    movl %ecx, %esp
; CHECK-NEXT:    retl # eh_return, addr: %ecx
entry:
  call void @llvm.eh.return.i32(i32 %a, ptr %b)
  unreachable
}

declare void @llvm.eh.return.i32(i32, i8*)
declare void @llvm.eh.unwind.init()
