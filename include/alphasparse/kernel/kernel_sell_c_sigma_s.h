#pragma once

#include "../spmat.h"

// alphasparseStatus_t add_s_sell_csigma(const spmat_sell_csigma_s_t *A, const float alpha, const spmat_sell_csigma_s_t *B, spmat_sell_csigma_s_t **C);
// alphasparseStatus_t add_s_sell_csigma_trans(const spmat_sell_csigma_s_t *A, const float alpha, const spmat_sell_csigma_s_t *B, spmat_sell_csigma_s_t **C);
// alphasparseStatus_t add_s_sell_csigma_conj(const spmat_sell_csigma_s_t *A, const float alpha, const spmat_sell_csigma_s_t *B, spmat_sell_csigma_s_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparseStatus_t gemv_s_sell_csigma(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*A^T*x + beta*y
// alphasparseStatus_t gemv_s_sell_csigma_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*A^T*x + beta*y
// alphasparseStatus_t gemv_s_sell_csigma_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);

// // alpha*(L+D+L')*x + beta*y
// alphasparseStatus_t symv_s_sell_csigma_n_lo(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(L+I+L')*x + beta*y
// alphasparseStatus_t symv_s_sell_csigma_u_lo(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U'+D+U)*x + beta*y
// alphasparseStatus_t symv_s_sell_csigma_n_hi(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U'+I+U)*x + beta*y
// alphasparseStatus_t symv_s_sell_csigma_u_hi(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);

// // alpha*(L+D)*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_n_lo(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(L+I)*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_u_lo(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U+D)*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_n_hi(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U+I)*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_u_hi(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);

// // alpha*(L+D)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_n_lo_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(L+I)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_u_lo_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U+D)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_n_hi_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U+I)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_u_hi_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);

// // alpha*(L+D)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_n_lo_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(L+I)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_u_lo_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U+D)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_n_hi_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*(U+I)^T*x + beta*y
// alphasparseStatus_t trmv_s_sell_csigma_u_hi_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);

// // alpha*D*x + beta*y
// alphasparseStatus_t diagmv_s_sell_csigma_n(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);
// // alpha*x + beta*y
// alphasparseStatus_t diagmv_s_sell_csigma_u(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const float beta, float *y);

// // --------------------------------------------------------------------------------------------------------------------------------

// // ---------------------------------------------------------------------------------------------------------------------------

// // alpha*A*B + beta*C
// alphasparseStatus_t gemm_s_sell_csigma_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alphasparseStatus_t gemm_s_sell_csigma_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*A^T*B + beta*C
// alphasparseStatus_t gemm_s_sell_csigma_row_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alphasparseStatus_t gemm_s_sell_csigma_col_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*A^T*B + beta*C
// alphasparseStatus_t gemm_s_sell_csigma_row_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alphasparseStatus_t gemm_s_sell_csigma_col_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*（L+D+L')^T*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_n_lo_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I+L')*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_u_lo_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U'+D+U)*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_n_hi_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U'+I+U)*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_u_hi_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*（L+D+L')^T*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_n_lo_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I+L')*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_u_lo_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U'+D+U)*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_n_hi_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U'+I+U)*B + beta*C
// alphasparseStatus_t symm_s_sell_csigma_u_hi_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*(L+D)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_lo_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_lo_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*op(U+D)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_hi_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*op(U+I)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_hi_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*(L+D)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_lo_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_lo_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+D)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_hi_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+I)*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_hi_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*(L+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_lo_row_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_lo_row_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_hi_row_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_hi_row_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_lo_col_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_lo_col_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_hi_col_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_hi_col_trans(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*(L+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_lo_row_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_lo_row_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_hi_row_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_hi_row_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_lo_col_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(L+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_lo_col_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+D)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_n_hi_col_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*(U+I)^T*B + beta*C
// alphasparseStatus_t trmm_s_sell_csigma_u_hi_col_conj(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // alpha*D*B + beta*C
// alphasparseStatus_t diagmm_s_sell_csigma_n_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*B + beta*C
// alphasparseStatus_t diagmm_s_sell_csigma_u_row(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*D*B + beta*C
// alphasparseStatus_t diagmm_s_sell_csigma_n_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// // alpha*B + beta*C
// alphasparseStatus_t diagmm_s_sell_csigma_u_col(const float alpha, const spmat_sell_csigma_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// // ---------------------------------------------------------------------------------------------------------------------------

// // -----------------------------------------------------------------------------------------------------

// // A*B
// alphasparseStatus_t spmmd_s_sell_csigma_row(const spmat_sell_csigma_s_t *matA, const spmat_sell_csigma_s_t *matB, float *C, const ALPHA_INT ldc);
// // A*B
// alphasparseStatus_t spmmd_s_sell_csigma_col(const spmat_sell_csigma_s_t *matA, const spmat_sell_csigma_s_t *matB, float *C, const ALPHA_INT ldc);
// // A^T*B
// alphasparseStatus_t spmmd_s_sell_csigma_row_trans(const spmat_sell_csigma_s_t *matA, const spmat_sell_csigma_s_t *matB, float *C, const ALPHA_INT ldc);
// // A^T*B
// alphasparseStatus_t spmmd_s_sell_csigma_col_trans(const spmat_sell_csigma_s_t *matA, const spmat_sell_csigma_s_t *matB, float *C, const ALPHA_INT ldc);

// // A^T*B
// alphasparseStatus_t spmmd_s_sell_csigma_row_conj(const spmat_sell_csigma_s_t *matA, const spmat_sell_csigma_s_t *matB, float *C, const ALPHA_INT ldc);
// // A^T*B
// alphasparseStatus_t spmmd_s_sell_csigma_col_conj(const spmat_sell_csigma_s_t *matA, const spmat_sell_csigma_s_t *matB, float *C, const ALPHA_INT ldc);

alphasparseStatus_t spmm_s_sell_csigma(const spmat_sell_csigma_s_t *A, const spmat_sell_csigma_s_t *B, spmat_sell_csigma_s_t **C);
// alphasparseStatus_t spmm_s_sell_csigma_trans(const spmat_sell_csigma_s_t *A, const spmat_sell_csigma_s_t *B, spmat_sell_csigma_s_t **C);
// alphasparseStatus_t spmm_s_sell_csigma_conj(const spmat_sell_csigma_s_t *A, const spmat_sell_csigma_s_t *B, spmat_sell_csigma_s_t **C);

// // -----------------------------------------------------------------------------------------------------

// // alpha*inv(L)*x
// alphasparseStatus_t trsv_s_sell_csigma_n_lo(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(L)*x
// alphasparseStatus_t trsv_s_sell_csigma_u_lo(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(U)*x
// alphasparseStatus_t trsv_s_sell_csigma_n_hi(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(U)*x
// alphasparseStatus_t trsv_s_sell_csigma_u_hi(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(L^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_n_lo_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(L^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_u_lo_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(U^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_n_hi_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(U^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_u_hi_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(L^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_n_lo_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(L^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_u_lo_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(U^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_n_hi_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*inv(U^T)*x
// alphasparseStatus_t trsv_s_sell_csigma_u_hi_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);

// // alpha*inv(D)*x
// alphasparseStatus_t diagsv_s_sell_csigma_n(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);
// // alpha*x
// alphasparseStatus_t diagsv_s_sell_csigma_u(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, float *y);

// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_lo_row(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_lo_row(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_hi_row(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_hi_row(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_lo_col(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_lo_col(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_hi_col(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_hi_col(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_lo_row_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_lo_row_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_hi_row_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_hi_row_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_lo_col_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_lo_col_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_hi_col_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_hi_col_trans(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_lo_row_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_lo_row_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_hi_row_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_hi_row_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_lo_col_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(L)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_lo_col_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_n_hi_col_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(U)*B
// alphasparseStatus_t trsm_s_sell_csigma_u_hi_col_conj(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

// // alpha*inv(D)*x
// alphasparseStatus_t diagsm_s_sell_csigma_n_row(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*x
// alphasparseStatus_t diagsm_s_sell_csigma_u_row(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*inv(D)*x
// alphasparseStatus_t diagsm_s_sell_csigma_n_col(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// // alpha*x
// alphasparseStatus_t diagsm_s_sell_csigma_u_col(const float alpha, const spmat_sell_csigma_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

// alphasparseStatus_t set_value_s_sell_csigma (spmat_sell_csigma_s_t * A, const ALPHA_INT row, const ALPHA_INT col, const float value);