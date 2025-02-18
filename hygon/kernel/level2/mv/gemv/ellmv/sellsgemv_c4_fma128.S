    .file	"ellmv_kernels.c"
    .text
    .globl	__spmv_sell_c4_serial_host_fma128_float
    .type	__spmv_sell_c4_serial_host_fma128_float, @function
__spmv_sell_c4_serial_host_fma128_float:
// void __spmv_sell_c4_serial_host_fma128_float(
// const float alpha:  xmm0
// const float beta :  xmm1
// const int num_chunks: edi
// const int *chunks_start: rsi
// const int *col_indices: rdx => rbx
// const float *values : rcx => r12
// const float *x :r8
// float *y : r9
// )
.L0:
    testl	%edi, %edi
    jle	.L11
    pushq	%r12
    pushq	%r13
    pushq	%r14
    pushq	%r15
    pushq	%rbp
    pushq	%rbx
    movq	%rdx, %rbx                // col_indices
    vshufps	$0, %xmm0, %xmm0, %xmm5 // alpha
    vshufps	$0, %xmm1, %xmm1, %xmm1 // beta
    leaq  (,%rdi,4), %rbp           // 行数  num_chunks * 4
    xorl	%edx, %edx
    xorq	%r15, %r15                 // chunk_id = 行 / 4
  // outer loop on chunks,

  /*
    rdx : 外层循环变量  行号 lrs,每层循环 + 4
    rbp : 外循环上界    行数 num_chunks * 4
    rdi : 内层循环变量 values地址
    r11 : 内层循环上界  下一列values地址 
  */
.L5:
    movslq	(%rsi,%r15,4), %r13   // chunk_start
    movslq	4(%rsi,%r15,4), %r10  // chunk_end
    leaq	(%rcx,%r13,4), %rdi     // &values[chunk_start]
    leaq	(%rbx,%r13,4), %rax     // &col_indices[chunk_start]

    leaq	(%rcx,%r10,4), %r11   // &values[chunk_end]
    cmpl	%r13d, %r10d          // width 等于 0?
    jle	.L6                     // 没有内层循环
    xorps	%xmm3, %xmm3       

    // movslq	(%rsi,%rdx), %r11 // chunk_start
    // movl	4(%rsi,%rdx), %r10d // chunk_end
    // movl	%r10d, %r12d       // chunk_end
    // subl	%r11d, %r10d        // nnz_chunk
    // sarl	$2, %r10d           // width
    // leaq	0(,%r11,4), %rdi    // chunk_start * 4
    // leaq	(%rbx,%rdi), %rax   // &col_indices[chunk_start]
    // testl	%r10d, %r10d        // width 等于 0?
    // jle	.L6                   // 没有内层循环
    // addq	%rcx, %rdi          // &values[chunk_start]
    // leaq	(%rcx,%r12,4), %r11 // 下一列values地址 values + chunk_end
    // xorps	%xmm3, %xmm3
  // inner loop on width
.L4:
    movslq	(%rax), %r10             // idx0
    movss	(%r8,%r10,4), %xmm0        // x[idx0]
    movslq	4(%rax), %r13            // idx1
    movss	(%r8,%r13,4), %xmm2        // x[idx1]
    shufps	$0, %xmm2, %xmm0
    movslq	8(%rax), %r14            // idx2
    movss	(%r8,%r14,4), %xmm4        // x[idx2]
    movslq	12(%rax), %r12           // idx3
    movss	(%r8,%r12,4), %xmm6        // x[idx3]
    shufps	$0, %xmm6, %xmm4
    shufps	$136, %xmm4, %xmm0
    vfmadd231ps	(%rdi), %xmm0, %xmm3 // acc += values[] * x[]
    addq	$16, %rdi                  // values  下一列
    addq	$16, %rax                  // indices 下一列
    cmpq	%rdi, %r11                 // rdi 是循环变量, r11是下一列 values 地址
    jne	.L4
.L3:
    vmulps	(%r9,%rdx,4), %xmm1, %xmm0  // y *= beta
    vfmadd132ps	%xmm5, %xmm0, %xmm3     // y += alpha * acc
    movups	%xmm3, (%r9,%rdx,4)         // store(y)
    addq	$4, %rdx                      // 跳四行
    incq %r15
    cmpq	%rdx, %rbp                    // 结束?
    jne	.L5
    popq	%rbx
    popq	%rbp
    popq	%r15
    popq	%r14
    popq	%r13
    popq	%r12
    ret
.L6:                                  // 防止当前行没有nnz
    vxorps	%xmm3, %xmm3, %xmm3
    jmp	.L3
.L11:
    ret
