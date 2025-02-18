#pragma once

#include "../spmat.h"

alphasparseStatus_t add_c_csc_plain(const spmat_csc_c_t *A, const ALPHA_Complex8 alpha, const spmat_csc_c_t *B, spmat_csc_c_t **C);
alphasparseStatus_t add_c_csc_trans_plain(const spmat_csc_c_t *A, const ALPHA_Complex8 alpha, const spmat_csc_c_t *B, spmat_csc_c_t **C);
alphasparseStatus_t add_c_csc_conj_plain(const spmat_csc_c_t *A, const ALPHA_Complex8 alpha, const spmat_csc_c_t *B, spmat_csc_c_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparseStatus_t gemv_c_csc_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t gemv_c_csc_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t gemv_c_csc_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t symv_c_csc_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t symv_c_csc_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t symv_c_csc_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t symv_c_csc_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t symv_c_csc_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t symv_c_csc_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t symv_c_csc_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t symv_c_csc_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t hermv_c_csc_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t hermv_c_csc_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t hermv_c_csc_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t hermv_c_csc_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')^T*x + beta*y
alphasparseStatus_t hermv_c_csc_n_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')^T*x + beta*y
alphasparseStatus_t hermv_c_csc_u_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)^T*x + beta*y
alphasparseStatus_t hermv_c_csc_n_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)^T*x + beta*y
alphasparseStatus_t hermv_c_csc_u_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)*x + beta*y
alphasparseStatus_t trmv_c_csc_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t trmv_c_csc_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t trmv_c_csc_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t trmv_c_csc_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_n_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_u_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_n_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_u_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D)^H*x + beta*y
alphasparseStatus_t trmv_c_csc_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^H*x + beta*y
alphasparseStatus_t trmv_c_csc_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^H*x + beta*y
alphasparseStatus_t trmv_c_csc_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^H*x + beta*y
alphasparseStatus_t trmv_c_csc_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);


// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t trmv_c_csc_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*D*x + beta*y
alphasparseStatus_t diagmv_c_csc_n_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*x + beta*y
alphasparseStatus_t diagmv_c_csc_u_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparseStatus_t gemm_c_csc_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparseStatus_t gemm_c_csc_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparseStatus_t gemm_c_csc_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparseStatus_t gemm_c_csc_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparseStatus_t gemm_c_csc_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparseStatus_t gemm_c_csc_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_c_csc_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_c_csc_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_c_csc_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_c_csc_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_c_csc_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_c_csc_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_c_csc_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_c_csc_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_c_csc_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_c_csc_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_c_csc_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_c_csc_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t symm_c_csc_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t symm_c_csc_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t symm_c_csc_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t symm_c_csc_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);


// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t hermm_c_csc_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t hermm_c_csc_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t hermm_c_csc_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t hermm_c_csc_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t hermm_c_csc_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t hermm_c_csc_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t hermm_c_csc_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t hermm_c_csc_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t hermm_c_csc_n_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t hermm_c_csc_u_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t hermm_c_csc_n_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t hermm_c_csc_u_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t hermm_c_csc_n_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t hermm_c_csc_u_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t hermm_c_csc_n_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t hermm_c_csc_u_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t trmm_c_csc_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t trmm_c_csc_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparseStatus_t trmm_c_csc_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparseStatus_t trmm_c_csc_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t trmm_c_csc_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t trmm_c_csc_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparseStatus_t trmm_c_csc_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparseStatus_t trmm_c_csc_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t trmm_c_csc_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparseStatus_t diagmm_c_csc_n_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t diagmm_c_csc_u_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparseStatus_t diagmm_c_csc_n_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t diagmm_c_csc_u_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparseStatus_t spmmd_c_csc_row_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A*B
alphasparseStatus_t spmmd_c_csc_col_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_c_csc_row_trans_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_c_csc_col_trans_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^H*B
alphasparseStatus_t spmmd_c_csc_row_conj_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^H*B
alphasparseStatus_t spmmd_c_csc_col_conj_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);


// A^T*B
alphasparseStatus_t spmmd_c_csc_row_conj_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t spmmd_c_csc_col_conj_plain(const spmat_csc_c_t *matA, const spmat_csc_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

alphasparseStatus_t spmm_c_csc_plain(const spmat_csc_c_t *A, const spmat_csc_c_t *B, spmat_csc_c_t **C);
alphasparseStatus_t spmm_c_csc_trans_plain(const spmat_csc_c_t *A, const spmat_csc_c_t *B, spmat_csc_c_t **C);
alphasparseStatus_t spmm_c_csc_conj_plain(const spmat_csc_c_t *A, const spmat_csc_c_t *B, spmat_csc_c_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparseStatus_t trsv_c_csc_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L)*x
alphasparseStatus_t trsv_c_csc_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparseStatus_t trsv_c_csc_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparseStatus_t trsv_c_csc_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_c_csc_n_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_c_csc_u_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_c_csc_n_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_c_csc_u_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_c_csc_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t trsv_c_csc_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_c_csc_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t trsv_c_csc_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(D)*x
alphasparseStatus_t diagsv_c_csc_n_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*x
alphasparseStatus_t diagsv_c_csc_u_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);


// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t trsm_c_csc_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t trsm_c_csc_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparseStatus_t diagsm_c_csc_n_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t diagsm_c_csc_u_row_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparseStatus_t diagsm_c_csc_n_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t diagsm_c_csc_u_col_plain(const ALPHA_Complex8 alpha, const spmat_csc_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

alphasparseStatus_t set_value_c_csc_plain (spmat_csc_c_t * A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex8 value);
