#pragma once

#include "../spmat.h"

alphasparseStatus_t add_z_bsr(const spmat_bsr_z_t *A, const ALPHA_Complex16 alpha, const spmat_bsr_z_t *B, spmat_bsr_z_t **C);
alphasparseStatus_t add_z_bsr_trans(const spmat_bsr_z_t *A, const ALPHA_Complex16 alpha, const spmat_bsr_z_t *B, spmat_bsr_z_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparseStatus_t gemv_z_bsr(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t gemv_z_bsr_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t gemv_z_bsr_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t symv_z_bsr_n_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t symv_z_bsr_u_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t symv_z_bsr_n_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t symv_z_bsr_u_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);


// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t symv_z_bsr_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t symv_z_bsr_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t symv_z_bsr_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t symv_z_bsr_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparseStatus_t hermv_z_bsr_n_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t hermv_z_bsr_u_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t hermv_z_bsr_n_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t hermv_z_bsr_u_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t hermv_z_bsr_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t hermv_z_bsr_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t hermv_z_bsr_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t hermv_z_bsr_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparseStatus_t trmv_z_bsr_n_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t trmv_z_bsr_u_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t trmv_z_bsr_n_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t trmv_z_bsr_u_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t trmv_z_bsr_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t trmv_z_bsr_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t trmv_z_bsr_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t trmv_z_bsr_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

alphasparseStatus_t trmv_z_bsr_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t trmv_z_bsr_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t trmv_z_bsr_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t trmv_z_bsr_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);


// alpha*D*x + beta*y
alphasparseStatus_t diagmv_z_bsr_n(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*x + beta*y
alphasparseStatus_t diagmv_z_bsr_u(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparseStatus_t gemm_z_bsr_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparseStatus_t gemm_z_bsr_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparseStatus_t gemm_z_bsr_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparseStatus_t gemm_z_bsr_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparseStatus_t gemm_z_bsr_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparseStatus_t gemm_z_bsr_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_z_bsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_z_bsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_z_bsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_z_bsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);


// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_z_bsr_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_z_bsr_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_z_bsr_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_z_bsr_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_z_bsr_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t hermm_z_bsr_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);


// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_z_bsr_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparseStatus_t diagmm_z_bsr_n_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t diagmm_z_bsr_u_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparseStatus_t diagmm_z_bsr_n_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t diagmm_z_bsr_u_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparseStatus_t spmmd_z_bsr_row(const spmat_bsr_z_t *matA, const spmat_bsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A*B
alphasparseStatus_t spmmd_z_bsr_col(const spmat_bsr_z_t *matA, const spmat_bsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_z_bsr_row_trans(const spmat_bsr_z_t *matA, const spmat_bsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_z_bsr_col_trans(const spmat_bsr_z_t *matA, const spmat_bsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_z_bsr_row_conj(const spmat_bsr_z_t *matA, const spmat_bsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_z_bsr_col_conj(const spmat_bsr_z_t *matA, const spmat_bsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

alphasparseStatus_t spmm_z_bsr(const spmat_bsr_z_t *A, const spmat_bsr_z_t *B, spmat_bsr_z_t **C);
alphasparseStatus_t spmm_z_bsr_trans(const spmat_bsr_z_t *A, const spmat_bsr_z_t *B, spmat_bsr_z_t **C);
alphasparseStatus_t spmm_z_bsr_conj(const spmat_bsr_z_t *A, const spmat_bsr_z_t *B, spmat_bsr_z_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparseStatus_t trsv_z_bsr_n_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L)*x
alphasparseStatus_t trsv_z_bsr_u_lo(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparseStatus_t trsv_z_bsr_n_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparseStatus_t trsv_z_bsr_u_hi(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_z_bsr_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_z_bsr_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_z_bsr_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_z_bsr_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_z_bsr_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_z_bsr_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_z_bsr_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_z_bsr_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(D)*x
alphasparseStatus_t diagsv_z_bsr_n(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*x
alphasparseStatus_t diagsv_z_bsr_u(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_z_bsr_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_z_bsr_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparseStatus_t diagsm_z_bsr_n_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t diagsm_z_bsr_u_row(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparseStatus_t diagsm_z_bsr_n_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t diagsm_z_bsr_u_col(const ALPHA_Complex16 alpha, const spmat_bsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparseStatus_t axpy_z(const ALPHA_INT nz,  const ALPHA_Complex16 a,  const ALPHA_Complex16* x,  const ALPHA_INT* indx,  ALPHA_Complex16* y);

alphasparseStatus_t gthr_z(const ALPHA_INT nz,	const ALPHA_Complex16* y, ALPHA_Complex16* x, const ALPHA_INT* indx);

alphasparseStatus_t rot_z(const ALPHA_INT nz, ALPHA_Complex16* x, const ALPHA_INT* indx, ALPHA_Complex16* y, const ALPHA_Complex16 c, const ALPHA_Complex16 s);

alphasparseStatus_t sctr_z(const ALPHA_INT nz, const ALPHA_Complex16* x, const ALPHA_INT* indx, ALPHA_Complex16* y);

ALPHA_Complex16 doti_z(const ALPHA_INT nz,  const ALPHA_Complex16* x,  const ALPHA_INT* indx, const ALPHA_Complex16* y);

alphasparseStatus_t set_value_z_bsr (spmat_bsr_z_t * A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex16 value);
alphasparseStatus_t update_values_z_bsr (spmat_bsr_z_t * A, const ALPHA_INT nvalues, const ALPHA_INT *indx, const ALPHA_INT *indy, ALPHA_Complex16 *values);
