	.file	"spmv_csr.c"
  .section .rodata
.LC0:
  .string "rows_start is %d, rows_end is %d, len is %d, alpha is %f, beta is %f\n"
.LC00:
  .string "number is %f,%f\n"
.LC01:
  .string "mov rows_end is %d\n\n"

	.text
  .globl __spmv_csr_serial_host_sse_complex_float
	.type	__spmv_csr_serial_host_sse_complex_float, @function
//regester used :
//register unused: %r15
//void __spmv_csr_serial_host_sse_complex_float(
//                            const Complex8 alpha,   // xmm0
//                            const Complex8 beta,    // xmm1
//                            const int num_rows,         //rdi
//                            const int * Ap, //row_start //rsi
//                            const int * Aj, //col_idx   //rdx 
//                            const Complex8 * Ax, //value   //rcx
//                            const Complex8 * x,    //r8
//                                  Complex8 * y)    //r9
// parameters:
__spmv_csr_serial_host_sse_complex_float:
.LFB2:
	movups	.LC1(%rip), %xmm4 # xmm4是 8 个 0xffffffff
.LCFI0:
	pushq	%r14
	movq	%rdx, %r14 // r14 = col_indx
	pushq	%r13
	movq	%rcx, %r13 // r13 = value
	pushq	%r12
	movq	%r9, %r12 // r12 = y
	pushq	%rbp
	movl	%edi, %ebp // ebp = num_rows
	pushq	%rbx
	movq	%rsi, %rbx  // rbx = rows_start
	xorl	%esi, %esi  // i 
	.p2align 4,,7

  movaps %xmm0, %xmm14 // [alpha.r, alpha.i]
  movaps %xmm1, %xmm15 // [beta.r, beta.i]
  vshufps $0x11,%xmm15,%xmm15,%xmm12 // [beta.i, beta.r]

.L17:
	cmpl	%ebp, %esi  // 和 sub %ebp, %esi 一样 (esi - ebp)
	jge	.L10
	movslq	%esi,%rax            //i:行号
	leaq	0(,%rax,4), %rdx       //i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]

	leaq	(%r12,%rdx,2), %rdi      // y[i] addr
  xorps %xmm8,%xmm8
  xorps %xmm10,%xmm10

  subl  $1, %ecx                  // unroll 2, ecx = rows_end[i] - 1,单精度循环展开2次
	movslq	%r10d,%rax              // row_start[i]

	leaq	(%r14,%rax,4), %rdx         // col_idx  addr
	leaq	(%r13,%rax,8), %r11         // value addr
	cmpl	%ecx, %r10d               // 
	jge	.L7
	.p2align 4,,7

// ########################main loop########################
// r14: col_idx
// r13: value
// r12 : y
// rdi : y + i
// r11 : value + i
// r10 : 循环变量
// r9 :?
// r8 : x
// rsi: current row, i
// rbp : rows
// rdx : col_idx + i
// rcx : rows_end[i] - 3
.L6:
 // 向量寄存器 [ar,ai,br,bi]: 左->右 == 低->高
 // x -> xmm2
 // values -> xmm6
  movups  (%r11), %xmm6        // value: a 
  addl   $2,  %r10d            // r10d 是循环变量(),    每次加 2
  addq  $16, %r11                    // r11  是&value[i],  每次 +16字节

  movslq (%rdx),%rax          // col_indx[0]
  movsd  (%r8,%rax,8),%xmm2   // x[col_indx[0]], 用movsd加载两个float到xmm2的低位
  movslq 0x4(%rdx),%rax        // col_indx[1]
  movhpd (%r8,%rax,8),%xmm2    // x[col_indx[1]], 用movsd加载两个float到xmm2的高位 : b
  addq   $8, %rdx              // rdx  是&col_idx[i],每次+8字节

  vshufps $0xb1,%xmm2,%xmm2,%xmm3 // TR(x): bSwap
  vshufps $0xf5,%xmm6,%xmm6,%xmm7    // Im(value): aIm
  vshufps $0xa0,%xmm6,%xmm6,%xmm9     // Re(value): aRe

  mulps %xmm7, %xmm3             // Im(value) * TR(x): aIm_bSwap
  vfmaddsub231ps %xmm9,%xmm2,%xmm3
  // 累加到xmm8上
  addps %xmm3,%xmm8

	cmpl	%r10d, %ecx     // ecx = nnz_row - 1
	jg	.L6
// ########################end loop########################
  // 规约
  movhlps %xmm8,%xmm10  //高位复数
  addps %xmm8,%xmm10    //两个复数实/虚部对应相加
.L7:
  // nnz个数加回来
  addl  $1, %ecx // ecx 现在是 rows_end[i]
  
.L8:
  // 如果算完了就跳至 .L4, r10d存的是已经遍历的nz个数
	cmpl	%r10d, %ecx          // rows_end[i] 等于 rows_start[i]就跳走,否则执行 .L9
  jle .L4
.L9:
  // subl %r10d, %ecx           // 还剩多少?
  // 否则 还剩一个nnz
  
	movslq	(%rdx),%rax        // col_idx
	movsd	(%r8,%rax,8), %xmm0  // x (a)
	movsd	(%r11), %xmm1        // value (b)
  vshufps $0xb1,%xmm1,%xmm1,%xmm3 // [ai, ar]
  vshufps $0xf5,%xmm0,%xmm0,%xmm7    // Im(value): aIm
  vshufps $0xa0,%xmm0,%xmm0,%xmm9     // Re(value): aRe
  mulps %xmm7, %xmm3             // Im(value) * TR(x): aIm_bSwap
  vfmaddsub231ps %xmm9,%xmm1,%xmm3
  addps %xmm3,%xmm10
.L4:
  movsd (%rdi),%xmm0              // y[i]
  vshufps $0xb1,%xmm15,%xmm15,%xmm13 
  vshufps $0xf5,%xmm0,%xmm0,%xmm7     
  vshufps $0xa0,%xmm0,%xmm0,%xmm9      
  mulps %xmm7, %xmm13              
  vfmaddsub231ps %xmm9,%xmm15,%xmm13 // y[i] * beta

	incl	%esi                        // 下一行
  
  vshufps $0xb1,%xmm14,%xmm14,%xmm12 
  vshufps $0xf5,%xmm10,%xmm10,%xmm6     
  vshufps $0xa0,%xmm10,%xmm10,%xmm8      
  mulps %xmm6, %xmm12              
  vfmaddsub231ps %xmm8,%xmm14,%xmm12 // alpha * xmm10
  addps %xmm13, %xmm12
	movsd	%xmm12, (%rdi)        // 写回y[i]
	jmp	.L17
.L10:
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.LFE5279:
	.size	__spmv_csr_serial_host_sse_complex_float, .-__spmv_csr_serial_host_sse_complex_float
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32

.FLOATCONST:
  .long 0x3F800000  // 1.0
  .long 0xBF800000  // -1.0
  .long 0x3F800000  // 1.0
  .long 0xBF800000  // -1.0
.LC1:
	.long	4294967295 # ffffffff
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.ident	"GCC: (GNU) 9.3.1 20200408 (Red Hat 9.3.1-2)"
	.section	.note.GNU-stack,"",@progbits
