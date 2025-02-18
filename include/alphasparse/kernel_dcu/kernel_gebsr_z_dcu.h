#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_add_z_gebsr(const spmat_gebsr_z_t *A, const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *B, spmat_gebsr_z_t **C);
alphasparseStatus_t dcu_add_z_gebsr_trans(const spmat_gebsr_z_t *A, const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *B, spmat_gebsr_z_t **C);
alphasparseStatus_t dcu_add_z_gebsr_conj(const spmat_gebsr_z_t *A, const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *B, spmat_gebsr_z_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparseStatus_t dcu_gemv_z_gebsr(alphasparseHandle_t          handle,
                                    alphasparse_layout_t       dir,
                                    ALPHA_INT             mb,
                                    ALPHA_INT             nb,
                                    ALPHA_INT             nnzb,
                                    const ALPHA_Complex16              alpha,
                                    const ALPHA_Complex16*              bsr_val,
                                    const ALPHA_INT*      bsr_row_ptr,
                                    const ALPHA_INT*      bsr_col_ind,
                                    ALPHA_INT             row_block_dim,
                                    ALPHA_INT             col_block_dim,
                                    const ALPHA_Complex16*              x,
                                    const ALPHA_Complex16              beta,
                                    ALPHA_Complex16*                    y);
// alpha*A^T*x + beta*y
alphasparseStatus_t dcu_gemv_z_gebsr_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t dcu_gemv_z_gebsr_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_n_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_u_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_n_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_u_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_gebsr_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_n_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_u_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_n_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_u_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+D+L')^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_gebsr_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_n_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_u_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_n_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_u_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_gebsr_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*D*x + beta*y
alphasparseStatus_t dcu_diagmv_z_gebsr_n(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*x + beta*y
alphasparseStatus_t dcu_diagmv_z_gebsr_u(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparseStatus_t dcu_gemm_z_gebsr(alphasparseHandle_t handle,
                                     alphasparse_layout_t dir,
                                     ALPHA_INT mb,
                                     ALPHA_INT n,
                                     ALPHA_INT kb,
                                     ALPHA_INT nnzb,
                                     const ALPHA_Complex16 alpha,
                                     const ALPHA_Complex16 *bsr_val,
                                     const ALPHA_INT *bsr_row_ptr,
                                     const ALPHA_INT *bsr_col_ind,
                                     ALPHA_INT row_block_dim,
                                     ALPHA_INT col_block_dim,
                                     const ALPHA_Complex16 *B,
                                     ALPHA_INT ldb,
                                     const ALPHA_Complex16 beta,
                                     ALPHA_Complex16 *C,
                                     ALPHA_INT ldc);

alphasparseStatus_t dcu_gemm_z_gebsr_transA(alphasparseHandle_t handle,
                                            alphasparse_layout_t dir,
                                            ALPHA_INT mb,
                                            ALPHA_INT n,
                                            ALPHA_INT kb,
                                            ALPHA_INT nnzb,
                                            const ALPHA_Complex16 alpha,
                                            const ALPHA_Complex16 *bsr_val,
                                            const ALPHA_INT *bsr_row_ptr,
                                            const ALPHA_INT *bsr_col_ind,
                                            ALPHA_INT row_block_dim,
                                            ALPHA_INT col_block_dim,
                                            const ALPHA_Complex16 *B,
                                            ALPHA_INT ldb,
                                            const ALPHA_Complex16 beta,
                                            ALPHA_Complex16 *C,
                                            ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_z_gebsr_transB(alphasparseHandle_t handle,
                                            alphasparse_layout_t dir,
                                            ALPHA_INT mb,
                                            ALPHA_INT n,
                                            ALPHA_INT kb,
                                            ALPHA_INT nnzb,
                                            const ALPHA_Complex16 alpha,
                                            const ALPHA_Complex16 *bsr_val,
                                            const ALPHA_INT *bsr_row_ptr,
                                            const ALPHA_INT *bsr_col_ind,
                                            ALPHA_INT row_block_dim,
                                            ALPHA_INT col_block_dim,
                                            const ALPHA_Complex16 *B,
                                            ALPHA_INT ldb,
                                            const ALPHA_Complex16 beta,
                                            ALPHA_Complex16 *C,
                                            ALPHA_INT ldc);

alphasparseStatus_t dcu_gemm_z_gebsr_transAB(alphasparseHandle_t handle,
                                             alphasparse_layout_t dir,
                                             ALPHA_INT mb,
                                             ALPHA_INT n,
                                             ALPHA_INT kb,
                                             ALPHA_INT nnzb,
                                             const ALPHA_Complex16 alpha,
                                             const ALPHA_Complex16 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT row_block_dim,
                                             ALPHA_INT col_block_dim,
                                             const ALPHA_Complex16 *B,
                                             ALPHA_INT ldb,
                                             const ALPHA_Complex16 beta,
                                             ALPHA_Complex16 *C,
                                             ALPHA_INT ldc); // alpha*A^T*B + beta*C
                                             
alphasparseStatus_t dcu_gemm_z_gebsr_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparseStatus_t dcu_gemm_z_gebsr_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_gebsr_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_gebsr_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_gebsr_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparseStatus_t dcu_diagmm_z_gebsr_n_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t dcu_diagmm_z_gebsr_u_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparseStatus_t dcu_diagmm_z_gebsr_n_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t dcu_diagmm_z_gebsr_u_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparseStatus_t dcu_spmmd_z_gebsr_row(const spmat_gebsr_z_t *matA, const spmat_gebsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A*B
alphasparseStatus_t dcu_spmmd_z_gebsr_col(const spmat_gebsr_z_t *matA, const spmat_gebsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_z_gebsr_row_trans(const spmat_gebsr_z_t *matA, const spmat_gebsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_z_gebsr_col_trans(const spmat_gebsr_z_t *matA, const spmat_gebsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

// A^T*B
alphasparseStatus_t dcu_spmmd_z_gebsr_row_conj(const spmat_gebsr_z_t *matA, const spmat_gebsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_z_gebsr_col_conj(const spmat_gebsr_z_t *matA, const spmat_gebsr_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

alphasparseStatus_t dcu_spmm_z_gebsr(const spmat_gebsr_z_t *A, const spmat_gebsr_z_t *B, spmat_gebsr_z_t **C);
alphasparseStatus_t dcu_spmm_z_gebsr_trans(const spmat_gebsr_z_t *A, const spmat_gebsr_z_t *B, spmat_gebsr_z_t **C);
alphasparseStatus_t dcu_spmm_z_gebsr_conj(const spmat_gebsr_z_t *A, const spmat_gebsr_z_t *B, spmat_gebsr_z_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparseStatus_t dcu_trsv_z_gebsr_n_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L)*x
alphasparseStatus_t dcu_trsv_z_gebsr_u_lo(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparseStatus_t dcu_trsv_z_gebsr_n_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparseStatus_t dcu_trsv_z_gebsr_u_hi(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_gebsr_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsv_z_gebsr_n(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*x
alphasparseStatus_t dcu_diagsv_z_gebsr_u(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_lo_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_hi_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_lo_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_hi_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_gebsr_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsm_z_gebsr_n_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t dcu_diagsm_z_gebsr_u_row(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsm_z_gebsr_n_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t dcu_diagsm_z_gebsr_u_col(const ALPHA_Complex16 alpha, const spmat_gebsr_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparseStatus_t dcu_set_value_z_gebsr(spmat_gebsr_z_t *A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex16 value);