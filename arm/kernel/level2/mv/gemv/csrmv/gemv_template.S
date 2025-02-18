	.arch armv8-a
	.file	"gemv_template.c"
	.text
	.align	2
	.p2align 3,,7
	.global	gemv_unroll_f32
	.type	gemv_unroll_f32, %function
gemv_unroll_f32:
.LFB3987:
	.cfi_startproc
	cmp	w0, 0
	add	x17, x1, 4
	add	x18, x3, 16
	mov	x16, 0
	ble	.L1
	.p2align 3,,7
.L2:
	ldr	w14, [x17, x16, lsl 2]
	ldr	w8, [x1, x16, lsl 2]
	sub	w6, w14, #3
	cmp	w8, w6
	bge	.L9
	sub	w15, w14, #4
	movi	v4.2s, #0
	sub	w15, w15, w8
	sxtw	x13, w8
	sbfiz	x7, x8, 2, 32
	lsr	w15, w15, 2
	add	x6, x3, x7
	fmov	s5, s4
	fmov	s3, s4
	add	x13, x13, x15, uxtw 2
	fmov	s2, s4
	add	x7, x2, x7
	add	x13, x18, x13, lsl 2
	.p2align 3,,7
.L4:
	ldpsw	x12, x11, [x7]
	ldp	s17, s19, [x6]
	ldpsw	x10, x9, [x7, 8]
	add	x7, x7, 16
	ldr	s7, [x4, x12, lsl 2]
	ldr	s18, [x4, x11, lsl 2]
	ldr	s16, [x4, x10, lsl 2]
	fmadd	s2, s17, s7, s2
	ldr	s6, [x4, x9, lsl 2]
	ldp	s17, s7, [x6, 8]
	add	x6, x6, 16
	fmadd	s3, s19, s18, s3
	cmp	x13, x6
	fmadd	s5, s17, s16, s5
	fmadd	s4, s7, s6, s4
	bne	.L4
	fadd	s3, s3, s5
	add	w6, w8, 4
	add	w8, w6, w15, lsl 2
	fadd	s3, s3, s4
.L3:
	cmp	w14, w8
	ble	.L5
	sxtw	x6, w8
	.p2align 3,,7
.L6:
	ldrsw	x7, [x2, x6, lsl 2]
	ldr	s5, [x3, x6, lsl 2]
	add	x6, x6, 1
	cmp	w14, w6
	ldr	s4, [x4, x7, lsl 2]
	fmadd	s2, s5, s4, s2
	bgt	.L6
.L5:
	fadd	s2, s2, s3
	ldr	s3, [x5, x16, lsl 2]
	fmul	s2, s2, s0
	fmadd	s2, s3, s1, s2
	str	s2, [x5, x16, lsl 2]
	add	x16, x16, 1
	cmp	w0, w16
	bgt	.L2
.L1:
	ret
	.p2align 2,,3
.L9:
	movi	v3.2s, #0
	fmov	s2, s3
	b	.L3
	.cfi_endproc
.LFE3987:
	.size	gemv_unroll_f32, .-gemv_unroll_f32
	.align	2
	.p2align 3,,7
	.global	gemv_3insert_sload_vhadd_f32
	.type	gemv_3insert_sload_vhadd_f32, %function
gemv_3insert_sload_vhadd_f32:
.LFB3988:
	.cfi_startproc
	cmp	w0, 0
	ble	.L15
	add	x17, x1, 4
	add	x18, x3, 16
	mov	x16, 0
	.p2align 3,,7
.L21:
	ldr	w14, [x17, x16, lsl 2]
	ldr	w7, [x1, x16, lsl 2]
	sub	w6, w14, #3
	cmp	w7, w6
	bge	.L22
	sub	w15, w14, #4
	sxtw	x13, w7
	sub	w15, w15, w7
	sbfiz	x6, x7, 2, 32
	movi	v3.4s, 0
	add	x8, x3, x6
	lsr	w15, w15, 2
	add	x6, x2, x6
	add	x13, x13, x15, uxtw 2
	add	x13, x18, x13, lsl 2
	.p2align 3,,7
.L18:

  // Acol
	ldpsw	x12, x11, [x6]
	ldpsw	x10, x9, [x6, 8]
	add	x6, x6, 16
  // x0,x1,x2,x3
	ldr	s2, [x4, x12, lsl 2]
	ldr	s7, [x4, x11, lsl 2]
	ldr	s6, [x4, x10, lsl 2]
	ldr	s5, [x4, x9, lsl 2]

	// ins	v2.s[0], v2.s[0]
  // Aval  
	ldr	q4, [x8], 16

	ins	v2.s[1], v7.s[0]
	cmp	x13, x8
	ins	v2.s[2], v6.s[0]
	ins	v2.s[3], v5.s[0]
	fmla	v3.4s, v2.4s, v4.4s
	bne	.L18
	add	w6, w7, 4
	add	w7, w6, w15, lsl 2
.L17:
	movi	v4.2s, #0
	cmp	w14, w7
	ble	.L19
	sxtw	x6, w7
	.p2align 3,,7
.L20:
	ldrsw	x7, [x2, x6, lsl 2]
	ldr	s6, [x3, x6, lsl 2]
	add	x6, x6, 1
	cmp	w14, w6
	ldr	s5, [x4, x7, lsl 2]
	fmadd	s4, s6, s5, s4
	bgt	.L20
.L19:
	faddp	v3.4s, v3.4s, v3.4s
	ldr	s5, [x5, x16, lsl 2]
	faddp	v3.4s, v3.4s, v3.4s
	fadd	s3, s3, s4
	fmul	s3, s3, s0
	fmadd	s3, s5, s1, s3
	str	s3, [x5, x16, lsl 2]
	add	x16, x16, 1
	cmp	w0, w16
	bgt	.L21
.L15:
	ret
	.p2align 2,,3
.L22:
	movi	v3.4s, 0
	b	.L17
	.cfi_endproc
.LFE3988:
	.size	gemv_3insert_sload_vhadd_f32, .-gemv_3insert_sload_vhadd_f32
	.align	2
	.p2align 3,,7
	.global	gemv_3insert_sload_revadd_f32
	.type	gemv_3insert_sload_revadd_f32, %function
gemv_3insert_sload_revadd_f32:
.LFB3989:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	cmp	w0, 0
	stp	x1, x2, [sp, 8]
	str	x3, [sp, 24]
	ble	.L27
	mov	x16, x1
	mov	x12, x3
	add	x17, x1, 4
	add	x18, x3, 16
	mov	x13, x2
	mov	x15, 0
	.p2align 3,,7
.L28:
	ldr	w11, [x17, x15, lsl 2]
	ldr	w2, [x16, x15, lsl 2]
	sub	w1, w11, #3
	cmp	w2, w1
	bge	.L35
	sub	w14, w11, #4
	sxtw	x10, w2
	sub	w14, w14, w2
	sbfiz	x1, x2, 2, 32
	movi	v3.4s, 0
	add	x3, x12, x1
	lsr	w14, w14, 2
	add	x1, x13, x1
	add	x10, x10, x14, uxtw 2
	add	x10, x18, x10, lsl 2
	.p2align 3,,7
.L30:
  // Acol
	ldpsw	x9, x8, [x1]
	ldpsw	x7, x6, [x1, 8]
	add	x1, x1, 16

  // x0,x1,x2,x3
	ldr	s2, [x4, x9, lsl 2]
	ldr	s7, [x4, x8, lsl 2]
	ldr	s6, [x4, x7, lsl 2]
	ldr	s5, [x4, x6, lsl 2]

  // Aval
	ldr	q4, [x3], 16

	cmp	x10, x3
	ins	v2.s[1], v7.s[0]
	ins	v2.s[2], v6.s[0]
	ins	v2.s[3], v5.s[0]
	fmla	v3.4s, v2.4s, v4.4s
	bne	.L30
	rev64	v4.4s, v3.4s
	add	w1, w2, 4
	add	w2, w1, w14, lsl 2
	fadd	v4.4s, v4.4s, v3.4s
	dup	v3.4s, v4.s[2]
	fadd	v3.4s, v3.4s, v4.4s
.L29:
	cmp	w11, w2
	ble	.L31
	movi	v4.2s, #0
	sxtw	x1, w2
	.p2align 3,,7
.L32:
	ldrsw	x2, [x13, x1, lsl 2]
	ldr	s6, [x12, x1, lsl 2]
	add	x1, x1, 1
	cmp	w11, w1
	ldr	s5, [x4, x2, lsl 2]
	fmadd	s4, s6, s5, s4
	bgt	.L32
	fadd	s3, s3, s4
.L31:
	ldr	s4, [x5, x15, lsl 2]
	fmul	s3, s0, s3
	fmadd	s3, s4, s1, s3
	str	s3, [x5, x15, lsl 2]
	add	x15, x15, 1
	cmp	w0, w15
	bgt	.L28
.L27:
	add	sp, sp, 32
	.cfi_remember_state
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L35:
	.cfi_restore_state
	movi	v3.2s, #0
	b	.L29
	.cfi_endproc
.LFE3989:
	.size	gemv_3insert_sload_revadd_f32, .-gemv_3insert_sload_revadd_f32
	.align	2
	.p2align 3,,7
	.global	gemv_4insert_sload_vhadd_f32
	.type	gemv_4insert_sload_vhadd_f32, %function
gemv_4insert_sload_vhadd_f32:
.LFB3990:
	.cfi_startproc
	cmp	w0, 0
	ble	.L41
	add	x17, x1, 4
	add	x18, x3, 16
	mov	x16, 0
	.p2align 3,,7
.L47:
	ldr	w14, [x17, x16, lsl 2]
	ldr	w7, [x1, x16, lsl 2]
	sub	w6, w14, #3
	cmp	w7, w6
	bge	.L48
	sub	w15, w14, #4
	sxtw	x13, w7
	sub	w15, w15, w7
	sbfiz	x6, x7, 2, 32
	movi	v3.4s, 0
	add	x8, x3, x6
	lsr	w15, w15, 2
	add	x6, x2, x6
	add	x13, x13, x15, uxtw 2
	add	x13, x18, x13, lsl 2
	.p2align 3,,7
.L44:
	ldpsw	x12, x11, [x6]
	ldpsw	x10, x9, [x6, 8]
	add	x6, x6, 16
	ldr	s4, [x4, x12, lsl 2]
	ldr	s7, [x4, x11, lsl 2]
	ldr	s6, [x4, x10, lsl 2]
	ins	v2.s[0], v4.s[0]
	ldr	s5, [x4, x9, lsl 2]
	ldr	q4, [x8], 16
	ins	v2.s[1], v7.s[0]
	cmp	x13, x8
	ins	v2.s[2], v6.s[0]
	ins	v2.s[3], v5.s[0]
	fmla	v3.4s, v2.4s, v4.4s
	bne	.L44
	add	w6, w7, 4
	add	w7, w6, w15, lsl 2
.L43:
	movi	v4.2s, #0
	cmp	w14, w7
	ble	.L45
	sxtw	x6, w7
	.p2align 3,,7
.L46:
	ldrsw	x7, [x2, x6, lsl 2]
	ldr	s6, [x3, x6, lsl 2]
	add	x6, x6, 1
	cmp	w14, w6
	ldr	s5, [x4, x7, lsl 2]
	fmadd	s4, s6, s5, s4
	bgt	.L46
.L45:
	faddp	v3.4s, v3.4s, v3.4s
	ldr	s5, [x5, x16, lsl 2]
	faddp	v3.4s, v3.4s, v3.4s
	fadd	s3, s3, s4
	fmul	s3, s3, s0
	fmadd	s3, s5, s1, s3
	str	s3, [x5, x16, lsl 2]
	add	x16, x16, 1
	cmp	w0, w16
	bgt	.L47
.L41:
	ret
	.p2align 2,,3
.L48:
	movi	v3.4s, 0
	b	.L43
	.cfi_endproc
.LFE3990:
	.size	gemv_4insert_sload_vhadd_f32, .-gemv_4insert_sload_vhadd_f32
	.align	2
	.p2align 3,,7
	.global	gemv_4insert_sload_revadd_f32
	.type	gemv_4insert_sload_revadd_f32, %function
gemv_4insert_sload_revadd_f32:
.LFB3991:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	cmp	w0, 0
	stp	x1, x2, [sp, 8]
	str	x3, [sp, 24]
	ble	.L53
	mov	x16, x1
	mov	x12, x3
	add	x17, x1, 4
	add	x18, x3, 16
	mov	x13, x2
	mov	x15, 0
	.p2align 3,,7
.L54:
	ldr	w11, [x17, x15, lsl 2]
	ldr	w2, [x16, x15, lsl 2]
	sub	w1, w11, #3
	cmp	w2, w1
	bge	.L61
	sub	w14, w11, #4
	sxtw	x10, w2
	sub	w14, w14, w2
	sbfiz	x1, x2, 2, 32
	movi	v3.4s, 0
	add	x3, x12, x1
	lsr	w14, w14, 2
	add	x1, x13, x1
	add	x10, x10, x14, uxtw 2
	add	x10, x18, x10, lsl 2
	.p2align 3,,7
.L56:
	ldpsw	x9, x8, [x1]
	ldpsw	x7, x6, [x1, 8]
	add	x1, x1, 16
	ldr	s4, [x4, x9, lsl 2]
	ldr	s7, [x4, x8, lsl 2]
	ldr	s6, [x4, x7, lsl 2]
	ins	v2.s[0], v4.s[0]
	ldr	s5, [x4, x6, lsl 2]
	ldr	q4, [x3], 16
	ins	v2.s[1], v7.s[0]
	cmp	x10, x3
	ins	v2.s[2], v6.s[0]
	ins	v2.s[3], v5.s[0]
	fmla	v3.4s, v2.4s, v4.4s
	bne	.L56
	rev64	v4.4s, v3.4s
	add	w1, w2, 4
	add	w2, w1, w14, lsl 2
	fadd	v4.4s, v4.4s, v3.4s
	dup	v3.4s, v4.s[2]
	fadd	v3.4s, v3.4s, v4.4s
.L55:
	cmp	w11, w2
	ble	.L57
	movi	v4.2s, #0
	sxtw	x1, w2
	.p2align 3,,7
.L58:
	ldrsw	x2, [x13, x1, lsl 2]
	ldr	s6, [x12, x1, lsl 2]
	add	x1, x1, 1
	cmp	w11, w1
	ldr	s5, [x4, x2, lsl 2]
	fmadd	s4, s6, s5, s4
	bgt	.L58
	fadd	s3, s3, s4
.L57:
	ldr	s4, [x5, x15, lsl 2]
	fmul	s3, s0, s3
	fmadd	s3, s4, s1, s3
	str	s3, [x5, x15, lsl 2]
	add	x15, x15, 1
	cmp	w0, w15
	bgt	.L54
.L53:
	add	sp, sp, 32
	.cfi_remember_state
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L61:
	.cfi_restore_state
	movi	v3.2s, #0
	b	.L55
	.cfi_endproc
.LFE3991:
	.size	gemv_4insert_sload_revadd_f32, .-gemv_4insert_sload_revadd_f32
  .align	2
	.p2align 3,,7
	.global	gemv_unroll16_3insert_sload_revadd_f32
	.type	gemv_unroll16_3insert_sload_revadd_f32, %function
gemv_unroll16_3insert_sload_revadd_f32:
	cmp	w0, 0
	ble	.Lu16_3ins69
	stp	x29, x30, [sp, -48]!
	add	x29, sp, 0
	stp	x19, x20, [sp, 16]
	sub	w20, w0, #1
	add	x20, x20, 1
	stp	x21, x22, [sp, 32]
	lsl	x20, x20, 2
	add	x21, x2, 64
	mov	x19, 0
	.p2align 2
.Lu16_3ins63:
	add	x0, x1, x19
	ldr	w6, [x1, x19]
	ldr	w30, [x0, 4]
	sub	w0, w30, #15
	cmp	w6, w0
	bge	.Lu16_3ins64
	movi	v7.4s, 0
	sub	w22, w30, w6
	sub	w22, w22, #16
	sxtw	x8, w6
	lsr	w22, w22, 4
	lsl	x0, x8, 2
	add	x8, x8, x22, uxtw 4
	add	x7, x3, x0
	add	x8, x21, x8, lsl 2
	add	x0, x2, x0
	orr	v6.16b, v7.16b, v7.16b
	orr	v17.16b, v7.16b, v7.16b
	orr	v16.16b, v7.16b, v7.16b
	.p2align 2
.Lu16_3ins60:
	ldpsw	x12, x18, [x0]
	add	x7, x7, 64
	ldpsw	x11, x17, [x0, 16]
	ldpsw	x10, x16, [x0, 32]
	ldpsw	x9, x15, [x0, 48]
	ldr	s2, [x4, x12, lsl 2]
	ldr	s3, [x4, x11, lsl 2]
	ldr	s4, [x4, x10, lsl 2]
	ldr	s5, [x4, x9, lsl 2]
	ldpsw	x14, x13, [x0, 8]
	ldpsw	x12, x11, [x0, 24]
	ldpsw	x10, x9, [x0, 40]
	// ins	v2.s[0], v21.s[0]
	ldr	s26, [x4, x15, lsl 2]
	// ins	v3.s[0], v20.s[0]
	ldr	s31, [x4, x18, lsl 2]
	// ins	v4.s[0], v19.s[0]
	ldr	s30, [x4, x17, lsl 2]
	// ins	v5.s[0], v18.s[0]
	ldr	s29, [x4, x16, lsl 2]
	ldr	q21, [x7, -64]
	ldr	q20, [x7, -48]
	ldr	q19, [x7, -32]
	ldr	q18, [x7, -16]
	ldr	s22, [x4, x10, lsl 2]
	ldr	s28, [x4, x14, lsl 2]
	ldr	s25, [x4, x13, lsl 2]
	ldr	s27, [x4, x12, lsl 2]
	ldr	s24, [x4, x11, lsl 2]
	ins	v5.s[1], v26.s[0]
	ldr	s23, [x4, x9, lsl 2]
	ldpsw	x10, x9, [x0, 56]
	add	x0, x0, 64
	ins	v2.s[1], v31.s[0]
	cmp	x8, x0
	ins	v3.s[1], v30.s[0]
	ins	v4.s[1], v29.s[0]
	ldr	s26, [x4, x10, lsl 2]
	ins	v2.s[2], v28.s[0]
	ins	v3.s[2], v27.s[0]
	ins	v4.s[2], v22.s[0]
	ldr	s22, [x4, x9, lsl 2]
	ins	v5.s[2], v26.s[0]
	ins	v2.s[3], v25.s[0]
	ins	v3.s[3], v24.s[0]
	ins	v4.s[3], v23.s[0]
	ins	v5.s[3], v22.s[0]
	fmla	v16.4s, v2.4s, v21.4s
	fmla	v17.4s, v3.4s, v20.4s
	fmla	v6.4s, v4.4s, v19.4s
	fmla	v7.4s, v5.4s, v18.4s
	bne	.Lu16_3ins60
	fadd	v6.4s, v6.4s, v7.4s
	add	w0, w6, 16
	add	w6, w0, w22, lsl 4
	fadd	v6.4s, v6.4s, v17.4s
	fadd	v6.4s, v6.4s, v16.4s
	rev64	v7.4s, v6.4s
	fadd	v7.4s, v7.4s, v6.4s
	dup	v6.4s, v7.s[2]
	fadd	v6.4s, v6.4s, v7.4s
.Lu16_3ins59:
	cmp	w30, w6
	ble	.Lu16_3ins61
	movi	v7.2s, #0
	sxtw	x0, w6
	.p2align 2
.Lu16_3ins62:
	ldrsw	x6, [x2, x0, lsl 2]
	ldr	s17, [x3, x0, lsl 2]
	add	x0, x0, 1
	cmp	w30, w0
	ldr	s16, [x4, x6, lsl 2]
	fmadd	s7, s17, s16, s7
	bgt	.Lu16_3ins62
	fadd	s6, s6, s7
.Lu16_3ins61:
	ldr	s7, [x5, x19]
	fmul	s6, s0, s6
	fmadd	s6, s7, s1, s6
	str	s6, [x5, x19]
	add	x19, x19, 4
	cmp	x20, x19
	bne	.Lu16_3ins63
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x29, x30, [sp], 48
	ret
	.p2align 3
.Lu16_3ins64:
	movi	v6.2s, #0
	b	.Lu16_3ins59
.Lu16_3ins69:
	ret
	.size	gemv_unroll16_3insert_sload_revadd_f32, .-gemv_unroll16_3insert_sload_revadd_f32

	.ident	"GCC: (GNU) 9.3.1 20200408 (Red Hat 9.3.1-2)"
	.section	.note.GNU-stack,"",@progbits
