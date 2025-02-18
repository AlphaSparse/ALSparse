    .file	"ellmv_kernels.c"
    .text
    .globl	__spmv_sell_c4_serial_host_fma256_double
    .type	__spmv_sell_c4_serial_host_fma256_double, @function
__spmv_sell_c4_serial_host_fma256_double:
// void __spmv_sell_c4_serial_host_fma256_double(
// const double alpha:  xmm0
// const double beta :  xmm1
// const int num_chunks: edi
// const int *chunks_start: rsi
// const int *col_indices: rdx => rbx
// const double *values : rcx => r12
// const double *x :r8
// double *y : r9
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
    vbroadcastsd	%xmm0, %ymm5 // alpha
    vbroadcastsd	%xmm1, %ymm11 // beta
    vmovapd	%ymm11, %ymm1 // beta
    sall  $2, %edi                   // num_chunks * 4
    movl %edi, %ebp                  // 行数  num_chunks * 4
    xorl	%edx, %edx
  // outer loop on chunks,

  /*
    rdx : 外层循环变量  行号 lrs,每层循环 + 4
    rbp : 外循环上界    行数 num_chunks * 4
    rdi : 内层循环变量 values地址
    r11 : 内层循环上界  下一列values地址
  */
.L5:
    movq %rdx,%rax
    sarq	$2, %rax              // chunk_id
    movslq	(%rsi,%rax,4), %r11 // chunk_start
    movl	4(%rsi,%rax,4), %r10d // chunk_end
    movl	%r10d, %r12d          // chunk_end
    subl	%r11d, %r10d          // nnz_chunk
    sarl	$2, %r10d             // width = nnz_chunk / 4
    leaq	0(,%r11,), %rdi       // chunk_start * 4
    leaq	(%rbx,%rdi,4), %rax   // &col_indices[chunk_start]
    testl	%r10d, %r10d          // width 等于 0?
    jle	.L6                     // 没有内层循环
    leaq	(%rcx,%rdi,8), %rdi   // &values[chunk_start]
    leaq	(%rcx,%r12,8), %r11   // &values[chunk_end]
    vxorpd	%ymm7, %ymm7,%ymm7        // y[0:7]

  // inner loop on width
.L4:
    movslq	(%rax), %r10             // idx0
    movsd	(%r8,%r10,8), %xmm0        // x[idx0]
    movslq	4(%rax), %r13            // idx1
    movhpd	(%r8,%r13,8), %xmm0        // x[idx1,idx0]
    movslq	8(%rax), %r14            // idx2
    movsd	(%r8,%r14,8), %xmm4        // x[idx2]
    movslq	12(%rax), %r15           // idx3
    movhpd	(%r8,%r15,8), %xmm4        // x[idx3,idx2]

    VINSERTF128 $1,%xmm4,%ymm0,%ymm0  // x[idx3,idx2,idx1,idx0]
    vfmadd231pd	(%rdi), %ymm0, %ymm7 // acc += values[] * x[]

    addq	$32, %rdi                  // values  下一列
    addq	$16, %rax                  // indices 下一列
    cmpq	%rdi, %r11                 // rdi 是循环变量, r11是下一列 values 地址
    jne	.L4
.L3:
    vmulpd	(%r9,%rdx,8), %ymm1, %ymm0   // y[0-7] *= beta
    vfmadd132pd	%ymm5, %ymm0, %ymm7      // y[0-7] += alpha * acc
    vmovupd	%ymm7, (%r9,%rdx,8)          // store(y[0-3])
    addq	$4, %rdx                       // 跳4行
    cmpq	%rdx, %rbp                     // 结束?
    jne	.L5
    popq	%rbx
    popq	%rbp
    popq	%r15
    popq	%r14
    popq	%r13
    popq	%r12
    ret
.L6:                                  // 防止当前行没有nnz
    vxorpd	%ymm7, %ymm7, %ymm7
    jmp	.L3
.L11:
    ret


.LS:
  .string "rdi %ld , r11 %ld row id is %ld\n"

.ONE:
  .string "rows is  %ld \n"