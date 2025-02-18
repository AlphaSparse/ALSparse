	.file	"spmv_csr.c"
  .section .rodata
.LC0:
  .string "rows_start is %d, rows_end is %d, len is %d, alpha is %f, beta is %f\n"
.LC00:
  .string "gather col_indx is %d,%d,%d,%d\n"
.LC01:
  .string "mov col_indx is %d\n\n"

	.text
  .globl __spmv_csr_serial_host_plain_float
	.type	__spmv_csr_serial_host_plain_float, @function
#regester used :
#register unused: %r15
#void __spmv_csr_serial_host_plain_float(
#                            const float alpha,   // xmm0
#                            const float beta,    // xmm1
#                            const int num_rows,         //rdi
#                            const int * Ap, //row_start //rsi
#                            const int * Aj, //col_idx   //rdx 
#                            const float * Ax, //value   //rcx
#                            const float * x,    //r8
#                                  float * y)    //r9
# parameters:
__spmv_csr_serial_host_plain_float:
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
	movl	%edi, %ebp #num_rows // ebp = num_rows

	pushq	%rbx
	movq	%rsi, %rbx  // rbx = rows_start

	xorl	%esi, %esi  // i 
	.p2align 4,,7

  movaps %xmm0, %xmm14 // alpha
  movaps %xmm1, %xmm15 // beta


.L17:
	cmpl	%ebp, %esi  // 和 sub %ebp, %esi 一样 (esi - ebp)
	jge	.L10
	movslq	%esi,%rax            //i:行号
	leaq	0(,%rax,4), %rdx       //i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]

	leaq	(%r12,%rdx), %rdi      // y[i] addr
  xorps %xmm8,%xmm8
  xorps %xmm9,%xmm9
  xorps %xmm10,%xmm10
  xorps %xmm11,%xmm11

  subl  $3, %ecx                  // unroll four, ecx = rows_end[i] - 3
	movslq	%r10d,%rax              // row_start[i]
	salq	$2, %rax                  // addr of row_start

	leaq	(%rax,%r14), %rdx         // col_idx  addr
	leaq	(%rax,%r13), %r11         // value addr
	cmpl	%ecx, %r10d               // 
	jge	.L7
	.p2align 4,,7

// ########################main loop########################
// r14: col_idx
// r13: value
// r12 : y
// rdi : y + i
// r11 : value + i
// r10 : row_start
// r9 :?
// r8 : x
// rsi: current row, i
// rbp : rows
// rdx : col_idx + i
// rcx : rows_end[i] - 3
.L6:
 // x -> xmm2
 // values -> xmm6
  movslq (%rdx),%rax          // col_indx[0]
  movslq 0x4(%rdx),%r9       // col_indx[1]

  movss  (%r8,%rax,4),%xmm0   // x[col_indx[0]]
  movss  (%r8,%r9,4),%xmm1   // x[col_indx[1]]

  movss  (%r11), %xmm4       // value[0]
  movss  4(%r11), %xmm5       // value[1]
  movss  8(%r11), %xmm6       // value[2]
  movss  12(%r11), %xmm7       // value[3]

  movslq 0x8(%rdx),%rax       // col_indx[2]
  movslq 0xc(%rdx),%r9       // col_indx[3]

  movss  (%r8,%rax,4),%xmm2   // x[col_indx[2]]
  movss  (%r8,%r9,4),%xmm3   // x[col_indx[3]]
  addl   $4,  %r10d           // r10d 是循环变量(),    每次加 4
  addq   $16, %rdx            // rdx  是&col_idx[i],每次+16
	
  vfmadd231ps	%xmm0, %xmm4, %xmm8 // xmm8是中间累加器, fma比mul+add慢===无语
  vfmadd231ps	%xmm1, %xmm5, %xmm9 // xmm8是中间累加器, fma比mul+add慢===无语
  vfmadd231ps	%xmm2, %xmm6, %xmm10 // xmm8是中间累加器, fma比mul+add慢===无语
  vfmadd231ps	%xmm3, %xmm7, %xmm11 // xmm8是中间累加器, fma比mul+add慢===无语

  // mulss %xmm0, %xmm4
  // mulss %xmm1, %xmm5
  // mulss %xmm2, %xmm6
  // mulss %xmm3, %xmm7

  // addss %xmm4,%xmm8
  // addss %xmm5,%xmm9
  // addss %xmm6,%xmm10
  // addss %xmm7,%xmm11


  addq  $16, %r11       // r11  是&value[i],  每次 +16
	cmpl	%r10d, %ecx     // ecx = nnz_row - 3
	jg	.L6
########################end loop########################
  #if 1
  addss %xmm8,%xmm9
  addss %xmm10,%xmm11
  addss %xmm9,%xmm11
  movaps %xmm11,%xmm10
  #endif
.L7:
  // nnz个数加回来
  addl  $3, %ecx // ecx 现在是 rows_end[i]
.L8:
  // 如果还没算完了就跳至 .L4, r10d存的是已经遍历的nz个数
	cmpl	%r10d, %ecx          // rows_end[i] 等于 rows_start[i]就跳走,否则执行 .L9
  jle .L4
.L9:
  subl %r10d, %ecx           // 还剩多少?
	movslq	(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0  // x[] 
	mulss	(%r11), %xmm0        // value
	addss	%xmm0, %xmm10        // xmm10 = sum
	cmpl	$1, %ecx           
	je	.L4                    //剩1个

  movslq	0x4(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	0x4(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$2, %ecx               
	je	.L4                       //剩2个

  movslq	0x8(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	0x8(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$3, %ecx              
	je	.L4                       //剩3个
.L4:
  vmulss (%rdi),%xmm15,%xmm1        // sum = y[i] * beta + 0 
	incl	%esi                        // 下一行
  vmulss	%xmm10, %xmm14,%xmm13           // sum * alpha
	addss	%xmm13, %xmm1           
  // vfmadd231ss %xmm10, %xmm14, %xmm1 // y[i] = sum * alpha + y[i] 
	movss	%xmm1, (%rdi)        // 写回y[i]
	jmp	.L17
.L10:
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.LFE5279:
	.size	__spmv_csr_serial_host_plain_float, .-__spmv_csr_serial_host_plain_float
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
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
