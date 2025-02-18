	.file	"spmv_csr.c"
  .section .rodata
.LC0:
  .string "rows_start is %d, rows_end is %d, len is %d, alpha is %f, beta is %f\n"
.LC00:
  .string "number is %f,%f\n"
.LC01:
  .string "mov rows_end is %d\n\n"

	.text
  .globl __spmv_csr_serial_host_sse_complex_double
	.type	__spmv_csr_serial_host_sse_complex_double, @function
//regester used :
//register unused: %r15
//void __spmv_csr_serial_host_sse_complex_double(
//                            const Complex16 alpha,   // xmm0: alpha.real, xmm1: alpha.imag
//                            const Complex16 beta,    // xmm2: alpha.real, xmm3: beta.imag
//                            const int num_rows,         //rdi
//                            const int * Ap, //row_start //rsi
//                            const int * Aj, //col_idx   //rdx 
//                            const Complex16 * Ax, //value   //rcx
//                            const Complex16 * x,    //r8
//                                  Complex16 * y)    //r9
// parameters:
__spmv_csr_serial_host_sse_complex_double:
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

  movlhps %xmm1, %xmm0  //[alpha.r,alpha.i]
  movapd %xmm0, %xmm14

  movlhps %xmm3, %xmm2  //[beta.r,beta.i]
  movapd %xmm2, %xmm15
.L17:
	cmpl	%ebp, %esi  // 和 sub %ebp, %esi 一样 (esi - ebp)
	jge	.L10
	movslq	%esi,%rax            //i:行号
	leaq	0(,%rax,4), %rdx       //i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]

	leaq	(%r12,%rdx,4), %rdi      // y+i 行*16 字节

  xorps %xmm8,%xmm8
  xorps %xmm12,%xmm12
  xorps %xmm10,%xmm10

	movslq	%r10d,%rax              // row_start[i]
	salq	$2, %rax                  // addr of row_start

	leaq	(%r14,%rax), %rdx          // col_idx  addr
	leaq	(%r13,%rax,4), %r11         // value addr
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
 // 向量寄存器 [ar,ai]: 左->右 == 低->高
 // x -> xmm2
 // values -> xmm6
  movslq (%rdx),%rax          // col_indx[0]
	salq	$4, %rax                  // addr of x[col_indx[0]]
  movupd  (%r11), %xmm6        // value: a 
  incl %r10d                   // r10d 是循环变量(),    每次++
  addq  $16, %r11              // r11  是&value[i],  每次 +16字节

  movupd  (%r8,%rax),%xmm2   // x[col_indx[0]], 加载两个double到xmm2的低位
  
  addq   $4, %rdx              // rdx  是&col_idx[i],每次+4字节

  vshufpd $0x5,%xmm2,%xmm2,%xmm3 // TR(x): bSwap
  vshufpd $0xf,%xmm6,%xmm6,%xmm7    // Im(value): aIm
  vshufpd $0x0,%xmm6,%xmm6,%xmm9     // Re(value): aRe

  mulpd %xmm7, %xmm3             // Im(value) * TR(x): aIm_bSwap
  vfmaddsub231pd %xmm9,%xmm2,%xmm3
  // 累加到xmm8上
  addpd %xmm3,%xmm8

	cmpl	%r10d, %ecx     // ecx = nnz_row - 1
	jg	.L6
// ########################end loop########################
  // 规约
  // movhlps %xmm8,%xmm10  //高位复数
  movapd %xmm8,%xmm10    
.L7:
.L4:
	incl	%esi                        // 下一行
  movupd (%rdi),%xmm0              // y[i]
  vshufpd $0x5,%xmm15,%xmm15,%xmm13 
  vshufpd $0xf,%xmm0,%xmm0,%xmm7     
  vshufpd $0x0,%xmm0,%xmm0,%xmm9      
  mulpd %xmm7, %xmm13              
  vfmaddsub231pd %xmm9,%xmm15,%xmm13 // y[i] * beta
  
  vshufpd $0x5,%xmm14,%xmm14,%xmm12 
  vshufpd $0xf,%xmm10,%xmm10,%xmm6     
  vshufpd $0x0,%xmm10,%xmm10,%xmm8      
  mulpd %xmm6, %xmm12              
  vfmaddsub231pd %xmm8,%xmm14,%xmm12 // alpha * xmm10
  addpd %xmm13, %xmm12
	movupd	%xmm12, (%rdi)        // 写回y[i]
	jmp	.L17
.L10:
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.LFE5279:
	.size	__spmv_csr_serial_host_sse_complex_double, .-__spmv_csr_serial_host_sse_complex_double
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
