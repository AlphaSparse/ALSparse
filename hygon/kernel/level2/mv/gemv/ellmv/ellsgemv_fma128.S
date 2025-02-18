    .text
    .globl	__spmv_ell_serial_host_sse_float
    .type	__spmv_ell_serial_host_sse_float, @function

// void __spmv_ell_serial_host_sse_float(const float alpha, // xmm0
//                                       const float beta,  // xmm1
//                                       const int lrs,     // edi
//                                       const int lre,     // esi
//                                       const int m,       // edx
//                                       const int width,   // ecx
//                                       const int *indices, //r8
//                                       const float *values, //r9
//                                       const float *x,      // rsp+40 ==> r10
//                                       float *y)            // rsp+48 ==> rbp
__spmv_ell_serial_host_sse_float:
.LFB5288:
// return adress : rsp + 32
    pushq	%r13 // rsp + 24
    pushq	%r12 // rsp + 16
    pushq	%rbp // rsp + 8
    pushq	%rbx // rsp
    movl	%esi, %ebx // lre
    movq	40(%rsp), %r10 // rsp + 40 : x
    movq	48(%rsp), %rbp // rsp + 48 : y
    cmpl	%esi, %edi  // r >= lre ? 外层循环一次都不用做?
    jge	.L8
    movl	%ecx, %esi // width
    movq	%r8, %r12  // indices
    movq	%r9, %r13  // values
    vshufps	$0, %xmm0, %xmm0, %xmm5 // [alpha,alpha,alpha,alpha]
    vshufps	$0, %xmm1, %xmm1, %xmm1 // [beta,beta,beta,beta]
    movslq	%edi, %r9               // lrs
    leaq	0(,%r9,4), %r8            // lrs * 4
    movslq	%edx, %rdx              // m
    leaq	0(,%rdx,4), %r11          // m * 4

  // **************main loop**************
  /* loop variable:
   * r8 : r * 4 (行: 外层循环)
   * r9 : r     (行: 外层循环)
   * ebx : lre
   *
   * edx : c    (列: 内层循环)
   * esi : width
   */

    // rbp : y
  // r13 : values
  // r12 : indices
    // r11 : m * 4
  // rdi : r13 + edx * r11 + r8 (values[]地址)
  // rax : r12 + edx * r11 + r8 (indices[]地址)
.L3:
    xorps	%xmm3, %xmm3      // accumulate
    testl	%esi, %esi        // width == 0 ?
    jle	.L6
    leaq	0(%r13,%r8), %rdi
    leaq	(%r12,%r8), %rax
    xorl	%edx, %edx
    xorps	%xmm3, %xmm3     // accumulate
  // **************inner loop************** ; loop variable : c : edx
.L4:
    movslq	(%rax), %rcx          // indices[0]
    movss	(%r10,%rcx,4), %xmm0    // x[0]
    movslq	4(%rax), %rcx         // indices[1]
    movss	(%r10,%rcx,4), %xmm2    // x[1]
    shufps	$0, %xmm2, %xmm0
    movslq	8(%rax), %rcx         // indices[2]
    movss	(%r10,%rcx,4), %xmm2  // x[2]
    movslq	12(%rax), %rcx        // indices[3]
    movss	(%r10,%rcx,4), %xmm4  // x[3]
    shufps	$0, %xmm4, %xmm2
    shufps	$136, %xmm2, %xmm0
    vfmadd231ps	(%rdi), %xmm0, %xmm3  // xmm3: a
    addl	$1, %edx
    addq	%r11, %rdi
    addq	%r11, %rax
    cmpl	%edx, %esi                  // c == width ?
    jne	.L4
.L6:
    vmulps	0(%rbp,%r8), %xmm1, %xmm0   // y[r] * beta
    vfmadd132ps	%xmm5, %xmm0, %xmm3     // alpha * accumlate + y[r] * beta
    movups	%xmm3, 0(%rbp,%r8)          // store => y[r]
    addq	$4, %r9
    addq	$16, %r8
    cmpl	%r9d, %ebx                    // r < rows ?
    jg	.L3
.L8:
    popq	%rbx
    popq	%rbp
    popq	%r12
    popq	%r13
    ret
.LFE5288:
    .size	__spmv_ell_serial_host_sse_float, .-__spmv_ell_serial_host_sse_float
    .ident	"GCC: (GNU) 9.3.1 20200408 (Red Hat 9.3.1-2)"
    .section	.note.GNU-stack,"",@progbits
