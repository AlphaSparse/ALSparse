#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_add_z_coo(const spmat_coo_z_t *A, const ALPHA_Complex16 alpha, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparseStatus_t dcu_add_z_coo_trans(const spmat_coo_z_t *A, const ALPHA_Complex16 alpha, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparseStatus_t dcu_add_z_coo_conj(const spmat_coo_z_t *A, const ALPHA_Complex16 alpha, const spmat_coo_z_t *B, spmat_coo_z_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparseStatus_t dcu_gemv_z_coo(alphasparseHandle_t handle,
                                    ALPHA_INT m,
                                    ALPHA_INT n,
                                    ALPHA_INT nnz,
                                    const ALPHA_Complex16 alpha,
                                    const ALPHA_Complex16 *coo_val,
                                    const ALPHA_INT *coo_row_ind,
                                    const ALPHA_INT *coo_col_ind,
                                    const ALPHA_Complex16 *x,
                                    const ALPHA_Complex16 beta,
                                    ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t dcu_gemv_z_coo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t dcu_gemv_z_coo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_symv_z_coo_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+D+L')^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)^T*x + beta*y
alphasparseStatus_t dcu_hermv_z_coo_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_z_coo_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*D*x + beta*y
alphasparseStatus_t dcu_diagmv_z_coo_n(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*x + beta*y
alphasparseStatus_t dcu_diagmv_z_coo_u(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT n,
                                        ALPHA_INT k,
                                        ALPHA_INT nnz,
                                        ALPHA_Complex16 alpha,
                                        const ALPHA_Complex16 *coo_val,
                                        const ALPHA_INT *coo_row_ind,
                                        const ALPHA_INT *coo_col_ind,
                                        const ALPHA_Complex16 *B,
                                        ALPHA_INT ldb,
                                        ALPHA_Complex16 beta,
                                        ALPHA_Complex16 *C,
                                        ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex16 alpha,
                                               const ALPHA_Complex16 *coo_val,
                                               const ALPHA_INT *coo_row_ind,
                                               const ALPHA_INT *coo_col_ind,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex16 beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex16 alpha,
                                               const ALPHA_Complex16 *coo_val,
                                               const ALPHA_INT *coo_row_ind,
                                               const ALPHA_INT *coo_col_ind,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex16 beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT k,
                                                ALPHA_INT nnz,
                                                ALPHA_Complex16 alpha,
                                                const ALPHA_Complex16 *coo_val,
                                                const ALPHA_INT *coo_row_ind,
                                                const ALPHA_INT *coo_col_ind,
                                                const ALPHA_Complex16 *B,
                                                ALPHA_INT ldb,
                                                ALPHA_Complex16 beta,
                                                ALPHA_Complex16 *C,
                                                ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_conjA(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT k,
                                              ALPHA_INT nnz,
                                              ALPHA_Complex16 alpha,
                                              const ALPHA_Complex16 *coo_val,
                                              const ALPHA_INT *coo_row_ind,
                                              const ALPHA_INT *coo_col_ind,
                                              const ALPHA_Complex16 *B,
                                              ALPHA_INT ldb,
                                              ALPHA_Complex16 beta,
                                              ALPHA_Complex16 *C,
                                              ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_conjB(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT k,
                                              ALPHA_INT nnz,
                                              ALPHA_Complex16 alpha,
                                              const ALPHA_Complex16 *coo_val,
                                              const ALPHA_INT *coo_row_ind,
                                              const ALPHA_INT *coo_col_ind,
                                              const ALPHA_Complex16 *B,
                                              ALPHA_INT ldb,
                                              ALPHA_Complex16 beta,
                                              ALPHA_Complex16 *C,
                                              ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_conjAB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex16 alpha,
                                               const ALPHA_Complex16 *coo_val,
                                               const ALPHA_INT *coo_row_ind,
                                               const ALPHA_INT *coo_col_ind,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex16 beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_transAconjB(alphasparseHandle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT k,
                                                    ALPHA_INT nnz,
                                                    ALPHA_Complex16 alpha,
                                                    const ALPHA_Complex16 *coo_val,
                                                    const ALPHA_INT *coo_row_ind,
                                                    const ALPHA_INT *coo_col_ind,
                                                    const ALPHA_Complex16 *B,
                                                    ALPHA_INT ldb,
                                                    ALPHA_Complex16 beta,
                                                    ALPHA_Complex16 *C,
                                                    ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_row_conjAtransB(alphasparseHandle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT k,
                                                    ALPHA_INT nnz,
                                                    ALPHA_Complex16 alpha,
                                                    const ALPHA_Complex16 *coo_val,
                                                    const ALPHA_INT *coo_row_ind,
                                                    const ALPHA_INT *coo_col_ind,
                                                    const ALPHA_Complex16 *B,
                                                    ALPHA_INT ldb,
                                                    ALPHA_Complex16 beta,
                                                    ALPHA_Complex16 *C,
                                                    ALPHA_INT ldc);

// alpha*A*B + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT n,
                                        ALPHA_INT k,
                                        ALPHA_INT nnz,
                                        ALPHA_Complex16 alpha,
                                        const ALPHA_Complex16 *coo_val,
                                        const ALPHA_INT *coo_row_ind,
                                        const ALPHA_INT *coo_col_ind,
                                        const ALPHA_Complex16 *B,
                                        ALPHA_INT ldb,
                                        ALPHA_Complex16 beta,
                                        ALPHA_Complex16 *C,
                                        ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex16 alpha,
                                               const ALPHA_Complex16 *coo_val,
                                               const ALPHA_INT *coo_row_ind,
                                               const ALPHA_INT *coo_col_ind,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex16 beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex16 alpha,
                                               const ALPHA_Complex16 *coo_val,
                                               const ALPHA_INT *coo_row_ind,
                                               const ALPHA_INT *coo_col_ind,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex16 beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT k,
                                                ALPHA_INT nnz,
                                                ALPHA_Complex16 alpha,
                                                const ALPHA_Complex16 *coo_val,
                                                const ALPHA_INT *coo_row_ind,
                                                const ALPHA_INT *coo_col_ind,
                                                const ALPHA_Complex16 *B,
                                                ALPHA_INT ldb,
                                                ALPHA_Complex16 beta,
                                                ALPHA_Complex16 *C,
                                                ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_conjA(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT k,
                                              ALPHA_INT nnz,
                                              ALPHA_Complex16 alpha,
                                              const ALPHA_Complex16 *coo_val,
                                              const ALPHA_INT *coo_row_ind,
                                              const ALPHA_INT *coo_col_ind,
                                              const ALPHA_Complex16 *B,
                                              ALPHA_INT ldb,
                                              ALPHA_Complex16 beta,
                                              ALPHA_Complex16 *C,
                                              ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_conjB(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT k,
                                              ALPHA_INT nnz,
                                              ALPHA_Complex16 alpha,
                                              const ALPHA_Complex16 *coo_val,
                                              const ALPHA_INT *coo_row_ind,
                                              const ALPHA_INT *coo_col_ind,
                                              const ALPHA_Complex16 *B,
                                              ALPHA_INT ldb,
                                              ALPHA_Complex16 beta,
                                              ALPHA_Complex16 *C,
                                              ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_conjAB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex16 alpha,
                                               const ALPHA_Complex16 *coo_val,
                                               const ALPHA_INT *coo_row_ind,
                                               const ALPHA_INT *coo_col_ind,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex16 beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_transAconjB(alphasparseHandle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT k,
                                                    ALPHA_INT nnz,
                                                    ALPHA_Complex16 alpha,
                                                    const ALPHA_Complex16 *coo_val,
                                                    const ALPHA_INT *coo_row_ind,
                                                    const ALPHA_INT *coo_col_ind,
                                                    const ALPHA_Complex16 *B,
                                                    ALPHA_INT ldb,
                                                    ALPHA_Complex16 beta,
                                                    ALPHA_Complex16 *C,
                                                    ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_z_coo_col_conjAtransB(alphasparseHandle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT k,
                                                    ALPHA_INT nnz,
                                                    ALPHA_Complex16 alpha,
                                                    const ALPHA_Complex16 *coo_val,
                                                    const ALPHA_INT *coo_row_ind,
                                                    const ALPHA_INT *coo_col_ind,
                                                    const ALPHA_Complex16 *B,
                                                    ALPHA_INT ldb,
                                                    ALPHA_Complex16 beta,
                                                    ALPHA_Complex16 *C,
                                                    ALPHA_INT ldc);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_z_coo_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_z_coo_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_z_coo_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparseStatus_t dcu_diagmm_z_coo_n_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t dcu_diagmm_z_coo_u_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparseStatus_t dcu_diagmm_z_coo_n_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t dcu_diagmm_z_coo_u_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparseStatus_t dcu_spmmd_z_coo_row(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A*B
alphasparseStatus_t dcu_spmmd_z_coo_col(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_z_coo_row_trans(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_z_coo_col_trans(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

// A^T*B
alphasparseStatus_t dcu_spmmd_z_coo_row_conj(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_z_coo_col_conj(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

alphasparseStatus_t dcu_spmm_z_coo(const spmat_coo_z_t *A, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparseStatus_t dcu_spmm_z_coo_trans(const spmat_coo_z_t *A, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparseStatus_t dcu_spmm_z_coo_conj(const spmat_coo_z_t *A, const spmat_coo_z_t *B, spmat_coo_z_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparseStatus_t dcu_trsv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L)*x
alphasparseStatus_t dcu_trsv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparseStatus_t dcu_trsv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparseStatus_t dcu_trsv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_coo_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_coo_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_coo_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_coo_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_coo_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_z_coo_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_coo_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_z_coo_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsv_z_coo_n(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*x
alphasparseStatus_t dcu_diagsv_z_coo_u(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_z_coo_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_z_coo_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsm_z_coo_n_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t dcu_diagsm_z_coo_u_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsm_z_coo_n_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t dcu_diagsm_z_coo_u_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparseStatus_t dcu_set_value_z_coo(spmat_coo_z_t *A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex16 value);