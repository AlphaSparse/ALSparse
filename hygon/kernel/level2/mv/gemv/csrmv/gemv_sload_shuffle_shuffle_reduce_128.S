	.text
  .globl __spmv_csr_serial_host_sse_float
	.type	__spmv_csr_serial_host_sse_float, @function
// regester used :
// register unused: %r15
// void __spmv_csr_serial_host_sse_float(
//                             const float alpha,   // xmm0
//                             const float beta,    // xmm1
//                             const int num_rows,         //rdi
//                             const int * Ap, //row_start //rsi
//                             const int * Aj, //col_idx   //rdx 
//                             const float * Ax, //value   //rcx
//                             const float * x,    //r8
//                                   float * y)    //r9
__spmv_csr_serial_host_sse_float:
.LFB2:
  // pushq %rax
  // pushq %rcx
  // pushq %rdx
  // pushq %rsi
  // pushq %rdi
  // pushq %r8
  // pushq %r9
  // pushq %r10
  // pushq %r11
  // cvtps2pd %xmm0,%xmm0
  // cvtps2pd %xmm1,%xmm1
  // movq %r10, %rsi // row_start
  // movq %rcx, %rdx // row_end
  // movl $0, %eax
  // sub %r10,%rcx // row_end - row_start
  // leaq .LC0(%rip), %rdi // 字符串""
  // call printf@PLT
  // popq %r11
  // popq %r10
  // popq %r9
  // popq %r8
  // popq %rdi
  // popq %rsi
  // popq %rdx
  // popq %rcx
  // popq %rax

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

// reg : c expression

/* loop var :
 * esi: 外循环变量 行号 i
 * ebp : rows
 * rdx : i * 4

 * r10d: 内循环变量 nz编号 ai
 * ecx : rows_end[i] - 3
 */

/* addr
 * 
 * r12 : y
 * rdi : y + i
 * r8: x
 * rbx : rows_start
 * r14 : col_indx
 * rdx : 当前col_idx. (col_idx + ai)
 * r13 : values
 * r11 : 当前values. (values + ai)
*/
	cmpl	%ebp, %esi  // 和 sub %ebp, %esi 一样 (esi - ebp)
	jge	.L10
	movslq	%esi,%rax            //i:行号
	leaq	0(,%rax,4), %rdx       //i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]

	leaq	(%r12,%rdx), %rdi      // y[i] addr
  xorps %xmm8,%xmm8
  xorps %xmm10,%xmm10

  subl  $3, %ecx                  // unroll four, ecx = rows_end[i] - 3
	movslq	%r10d,%rax              // row_start[i]
	salq	$2, %rax                  // addr of row_start

	leaq	(%rax,%r14), %rdx         // col_idx  addr
	leaq	(%rax,%r13), %r11         // value addr
  // prefetchnta (%rdx) //prfm col_idx 
  // prefetchnta (%r11) //prfm value
	cmpl	%ecx, %r10d               // 
	jge	.L7
	.p2align 4,,7

// ########################inner loop########################
.L6:
 // x -> xmm2
 // values -> xmm6

  movslq (%rdx),%rax          // col_indx[0]
  movss  (%r8,%rax,4),%xmm2   // x[col_indx[0]]
  movslq 0x4(%rdx),%rax       // col_indx[1]
  movss  (%r8,%rax,4),%xmm0   // x[col_indx[1]]
  movslq 0x8(%rdx),%rax       // col_indx[2]
  shufps $0x0,%xmm0,%xmm2    
  addl   $4,  %r10d           // r10d 是循环变量(),    每次加 4
  movups  (%r11), %xmm6       // value
  movss  (%r8,%rax,4),%xmm3   // x[col_indx[2]]
  movslq 0xc(%rdx),%rax       // col_indx[3]
  movss  (%r8,%rax,4),%xmm4   // x[col_indx[3]]
  shufps $0x0,%xmm4,%xmm3 
  addq   $16, %rdx            // rdx  是&col_idx[i],每次+16
  shufps $0x88,%xmm3,%xmm2

	vfmadd231ps	%xmm6, %xmm2, %xmm8 // xmm8是中间累加器, 
  addq  $16, %r11       // r11  是&value[i],  每次 +16
	cmpl	%r10d, %ecx     // ecx = nnz_row - 3
	jg	.L6
// ########################end loop########################
  // 121-128 行做xmm8规约,结果放在 xmm10
  #if 1
  movaps    %xmm8, %xmm9                                 
  shufps    $0xb1,%xmm8,%xmm9
  addps     %xmm8,%xmm9
  movaps    %xmm9,%xmm10
  unpckhps  %xmm9,%xmm10
  addss     %xmm9,%xmm10
  #endif
.L7:
  subl %r10d, %ecx           // 跳出来的时候 ecx + 3 - r10d 即剩余的nnz个数字, 可能值是 0,1,2,3; 即 ecx - r10d 可能值是 -3, -2, -1, 0
  cmpl	$-3, %ecx           
	jle	.L4                    //剩0个

	movslq	(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0  // x[] 
  vfmadd231ss (%r11), %xmm0, %xmm10 // val[0] * x[0] + accu

  cmpl	$-2, %ecx           
	jle	.L4                      //剩1个

  movslq	0x4(%rdx),%r9        // col_idx[jj]
	movss	(%r8,%r9,4), %xmm2     // x[] 
  vfmadd231ss 0x4(%r11), %xmm2, %xmm10 // val[1] * x[1] + accu
	cmpl	$-1, %ecx               
	jle	.L4                      //剩2个

  movslq	0x8(%rdx),%rcx       // col_idx[jj]
	movss	(%r8,%rcx,4), %xmm3    // x[] 
  vfmadd231ss 0x8(%r11), %xmm3, %xmm10 // val[2] * x[2] + accu

.L4:
  vmulss (%rdi),%xmm15,%xmm1        // y[i] * beta
	incl	%esi                        // 下一行
  vfmadd231ss %xmm10, %xmm14, %xmm1 // y[i] = sum * alpha + y[i] 
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
	.size	__spmv_csr_serial_host_sse_float, .-__spmv_csr_serial_host_sse_float
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC1:
	.ident	"GCC: (GNU) 9.3.1 20200408 (Red Hat 9.3.1-2)"
	.section	.note.GNU-stack,"",@progbits
