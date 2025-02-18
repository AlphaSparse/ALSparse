	.globl	__spmv_bsr4x1_serial_host_sse_float
	.type	__spmv_bsr4x1_serial_host_sse_float, @function

// void __spmv_bsr4x1_serial_host_sse_float(
//  const float alpha,          xmm0
//  const float beta,           xmm1
//  const int num_chunks,       rdi
//  const int *chunks_start,    rsi => r10
//  const int *col_indices,     rdx
//  const float *values,        rcx
//  const float *x,             r8
//  float *y)                   r9
__spmv_bsr4x1_serial_host_sse_float:
.START:
	testl	%edi, %edi
	jle	.L11
	pushq	%rbx
	movq	%rsi, %r10
	salq	$2, %rdi                  // num_chunks * 4
  leaq (%r9,%rdi,4), %rdi         // y + num_chunks * 4
  vshufps	$0, %xmm0, %xmm0, %xmm5 // [alpha,alpha,alpha,alpha]
  vshufps	$0, %xmm1, %xmm1, %xmm1 // [beta,beta,beta,beta]

.L5:
// outer loop

/*
  外循环变量 r9 : y  , 每次迭代 + 16字节
  外循环变量 r10 : chunk_start  , 每次迭代 + 4字节
  外循环上界 rdi : y 最后一个元素地址

  内循环变量 rax : nnz id (ai), 一个ai对应4个nz
  内循环上界 rbx : rows_end[r9]

*/
	xorps	%xmm7, %xmm7     // accum
	movslq	(%r10), %rax       // rows_start[]
	movl	4(%r10), %ebx        // rows_end[]
	cmpl	%ebx, %eax
	jge	.L3
  
.L4:
  movq %rax,%rsi
  salq $2,%rsi
	movslq	(%rdx,%rax,4), %r11   // idx0
	movss	(%r8,%r11,4), %xmm2     // x[idx0]
  shufps $0, %xmm2, %xmm2       // x[idx0,idx0,idx0,idx0]
	vfmadd231ps	(%rcx,%rsi,4), %xmm2, %xmm7  // values[] * x[]
  incq %rax                    // ai++
	cmpl	%eax, %ebx
	jg	.L4
.L3:
  vmulps	0(%r9), %xmm1, %xmm0   // y[r] * beta
  vfmadd132ps	%xmm5, %xmm0, %xmm7     // alpha * accumlate + y[r] * beta
  movups	%xmm7, 0(%r9)          // store => y[r]
	addq	$4, %r10
	addq	$16, %r9
	cmpq	%r9, %rdi
	jne	.L5
	popq	%rbx
	ret
.L11:
	ret