#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_add_c_csr(const spmat_csr_c_t *A, const ALPHA_Complex8 alpha, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparseStatus_t dcu_add_c_csr_trans(const spmat_csr_c_t *A, const ALPHA_Complex8 alpha, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparseStatus_t dcu_add_c_csr_conj(const spmat_csr_c_t *A, const ALPHA_Complex8 alpha, const spmat_csr_c_t *B, spmat_csr_c_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparseStatus_t dcu_gemv_c_csr(alphasparseHandle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT nnz,
                                   const ALPHA_Complex8 alpha,
                                   const ALPHA_Complex8 *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   alphasparse_dcu_mat_info_t info,
                                   const ALPHA_Complex8 *x,
                                   const ALPHA_Complex8 beta,
                                   ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t dcu_gemv_c_csr_trans(alphasparseHandle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT nnz,
                                   const ALPHA_Complex8 alpha,
                                   const ALPHA_Complex8 *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   alphasparse_dcu_mat_info_t info,
                                   const ALPHA_Complex8 *x,
                                   const ALPHA_Complex8 beta,
                                   ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparseStatus_t dcu_gemv_c_csr_conj(alphasparseHandle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT nnz,
                                   const ALPHA_Complex8 alpha,
                                   const ALPHA_Complex8 *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   alphasparse_dcu_mat_info_t info,
                                   const ALPHA_Complex8 *x,
                                   const ALPHA_Complex8 beta,
                                   ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_n_lo(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_u_lo(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_n_hi(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_u_hi(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_symv_c_csr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_n_lo(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_u_lo(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_n_hi(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_u_hi(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D+L')^T*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')^T*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)^T*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)^T*x + beta*y
alphasparseStatus_t dcu_hermv_c_csr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_n_lo(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_u_lo(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_n_hi(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_u_hi(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparseStatus_t dcu_trmv_c_csr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*D*x + beta*y
alphasparseStatus_t dcu_diagmv_c_csr_n(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*x + beta*y
alphasparseStatus_t dcu_diagmv_c_csr_u(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row(alphasparseHandle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT k,
                                   ALPHA_INT nnz,
                                   ALPHA_Complex8 alpha,
                                   const ALPHA_Complex8 *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   const ALPHA_Complex8 *B,
                                   ALPHA_INT ldb,
                                   ALPHA_Complex8 beta,
                                   ALPHA_Complex8 *C,
                                   ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_transA(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 *B,
                                          ALPHA_INT ldb,
                                          ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_transB(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 *B,
                                          ALPHA_INT ldb,
                                          ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_transAB(alphasparseHandle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           ALPHA_Complex8 alpha,
                                           const ALPHA_Complex8 *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const ALPHA_Complex8 *B,
                                           ALPHA_INT ldb,
                                           ALPHA_Complex8 beta,
                                           ALPHA_Complex8 *C,
                                           ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_conjA(alphasparseHandle_t handle,
                                         ALPHA_INT m,
                                         ALPHA_INT n,
                                         ALPHA_INT k,
                                         ALPHA_INT nnz,
                                         ALPHA_Complex8 alpha,
                                         const ALPHA_Complex8 *csr_val,
                                         const ALPHA_INT *csr_row_ptr,
                                         const ALPHA_INT *csr_col_ind,
                                         const ALPHA_Complex8 *B,
                                         ALPHA_INT ldb,
                                         ALPHA_Complex8 beta,
                                         ALPHA_Complex8 *C,
                                         ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_conjB(alphasparseHandle_t handle,
                                         ALPHA_INT m,
                                         ALPHA_INT n,
                                         ALPHA_INT k,
                                         ALPHA_INT nnz,
                                         ALPHA_Complex8 alpha,
                                         const ALPHA_Complex8 *csr_val,
                                         const ALPHA_INT *csr_row_ptr,
                                         const ALPHA_INT *csr_col_ind,
                                         const ALPHA_Complex8 *B,
                                         ALPHA_INT ldb,
                                         ALPHA_Complex8 beta,
                                         ALPHA_Complex8 *C,
                                         ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_conjAB(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 *B,
                                          ALPHA_INT ldb,
                                          ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_transAconjB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               const ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex8 beta,
                                               ALPHA_Complex8 *C,
                                               ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_row_conjAtransB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               const ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex8 beta,
                                               ALPHA_Complex8 *C,
                                               ALPHA_INT ldc);

// alpha*A*B + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col(alphasparseHandle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT k,
                                   ALPHA_INT nnz,
                                   ALPHA_Complex8 alpha,
                                   const ALPHA_Complex8 *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   const ALPHA_Complex8 *B,
                                   ALPHA_INT ldb,
                                   ALPHA_Complex8 beta,
                                   ALPHA_Complex8 *C,
                                   ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_transA(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 *B,
                                          ALPHA_INT ldb,
                                          ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_transB(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 *B,
                                          ALPHA_INT ldb,
                                          ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_transAB(alphasparseHandle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           ALPHA_Complex8 alpha,
                                           const ALPHA_Complex8 *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const ALPHA_Complex8 *B,
                                           ALPHA_INT ldb,
                                           ALPHA_Complex8 beta,
                                           ALPHA_Complex8 *C,
                                           ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_conjA(alphasparseHandle_t handle,
                                         ALPHA_INT m,
                                         ALPHA_INT n,
                                         ALPHA_INT k,
                                         ALPHA_INT nnz,
                                         ALPHA_Complex8 alpha,
                                         const ALPHA_Complex8 *csr_val,
                                         const ALPHA_INT *csr_row_ptr,
                                         const ALPHA_INT *csr_col_ind,
                                         const ALPHA_Complex8 *B,
                                         ALPHA_INT ldb,
                                         ALPHA_Complex8 beta,
                                         ALPHA_Complex8 *C,
                                         ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_conjB(alphasparseHandle_t handle,
                                         ALPHA_INT m,
                                         ALPHA_INT n,
                                         ALPHA_INT k,
                                         ALPHA_INT nnz,
                                         ALPHA_Complex8 alpha,
                                         const ALPHA_Complex8 *csr_val,
                                         const ALPHA_INT *csr_row_ptr,
                                         const ALPHA_INT *csr_col_ind,
                                         const ALPHA_Complex8 *B,
                                         ALPHA_INT ldb,
                                         ALPHA_Complex8 beta,
                                         ALPHA_Complex8 *C,
                                         ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_conjAB(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 *B,
                                          ALPHA_INT ldb,
                                          ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_transAconjB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               const ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex8 beta,
                                               ALPHA_Complex8 *C,
                                               ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemm_c_csr_col_conjAtransB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               ALPHA_INT nnz,
                                               ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               const ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               ALPHA_Complex8 beta,
                                               ALPHA_Complex8 *C,
                                               ALPHA_INT ldc);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_symm_c_csr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparseStatus_t dcu_hermm_c_csr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparseStatus_t dcu_trmm_c_csr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparseStatus_t dcu_diagmm_c_csr_n_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t dcu_diagmm_c_csr_u_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparseStatus_t dcu_diagmm_c_csr_n_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparseStatus_t dcu_diagmm_c_csr_u_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparseStatus_t dcu_gemmi_c_csr(alphasparseHandle_t handle,
                                    ALPHA_INT m,
                                    ALPHA_INT n,
                                    ALPHA_INT k,
                                    ALPHA_INT nnz,
                                    const ALPHA_Complex8 alpha,
                                    const ALPHA_Complex8 *A,
                                    ALPHA_INT lda,
                                    const ALPHA_Complex8 *csr_val,
                                    const ALPHA_INT *csr_row_ptr,
                                    const ALPHA_INT *csr_col_ind,
                                    const ALPHA_Complex8 beta,
                                    ALPHA_Complex8 *C,
                                    ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_transA(alphasparseHandle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           const ALPHA_Complex8 alpha,
                                           const ALPHA_Complex8 *A,
                                           ALPHA_INT lda,
                                           const ALPHA_Complex8 *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const ALPHA_Complex8 beta,
                                           ALPHA_Complex8 *C,
                                           ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_transB(alphasparseHandle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           const ALPHA_Complex8 alpha,
                                           const ALPHA_Complex8 *A,
                                           ALPHA_INT lda,
                                           const ALPHA_Complex8 *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const ALPHA_Complex8 beta,
                                           ALPHA_Complex8 *C,
                                           ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_transAB(alphasparseHandle_t handle,
                                            ALPHA_INT m,
                                            ALPHA_INT n,
                                            ALPHA_INT k,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex8 alpha,
                                            const ALPHA_Complex8 *A,
                                            ALPHA_INT lda,
                                            const ALPHA_Complex8 *csr_val,
                                            const ALPHA_INT *csr_row_ptr,
                                            const ALPHA_INT *csr_col_ind,
                                            const ALPHA_Complex8 beta,
                                            ALPHA_Complex8 *C,
                                            ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_conjA(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          const ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *A,
                                          ALPHA_INT lda,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_conjB(alphasparseHandle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          const ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *A,
                                          ALPHA_INT lda,
                                          const ALPHA_Complex8 *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_conjAB(alphasparseHandle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           const ALPHA_Complex8 alpha,
                                           const ALPHA_Complex8 *A,
                                           ALPHA_INT lda,
                                           const ALPHA_Complex8 *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const ALPHA_Complex8 beta,
                                           ALPHA_Complex8 *C,
                                           ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_transAconjB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT k,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *A,
                                                ALPHA_INT lda,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                const ALPHA_Complex8 beta,
                                                ALPHA_Complex8 *C,
                                                ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparseStatus_t dcu_gemmi_c_csr_conjAtransB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT k,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *A,
                                                ALPHA_INT lda,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                const ALPHA_Complex8 beta,
                                                ALPHA_Complex8 *C,
                                                ALPHA_INT ldc);

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparseStatus_t dcu_spmmd_c_csr_row(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A*B
alphasparseStatus_t dcu_spmmd_c_csr_col(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_c_csr_row_trans(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_c_csr_col_trans(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

// A^T*B
alphasparseStatus_t dcu_spmmd_c_csr_row_conj(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparseStatus_t dcu_spmmd_c_csr_col_conj(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

alphasparseStatus_t dcu_spmm_c_csr(const spmat_csr_c_t *A, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparseStatus_t dcu_spmm_c_csr_trans(const spmat_csr_c_t *A, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparseStatus_t dcu_spmm_c_csr_conj(const spmat_csr_c_t *A, const spmat_csr_c_t *B, spmat_csr_c_t **C);

// -----------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------
// C = alpha * A * B + beta * D
alphasparseStatus_t dcu_spgemm_c_csr(alphasparseHandle_t handle,
                                     ALPHA_INT m,
                                     ALPHA_INT n,
                                     ALPHA_INT k,
                                     const ALPHA_Complex8 alpha,
                                     ALPHA_INT nnz_A,
                                     const ALPHA_Complex8 *csr_val_A,
                                     const ALPHA_INT *csr_row_ptr_A,
                                     const ALPHA_INT *csr_col_ind_A,
                                     ALPHA_INT nnz_B,
                                     const ALPHA_Complex8 *csr_val_B,
                                     const ALPHA_INT *csr_row_ptr_B,
                                     const ALPHA_INT *csr_col_ind_B,
                                     const ALPHA_Complex8 beta,
                                     ALPHA_INT nnz_D,
                                     const ALPHA_Complex8 *csr_val_D,
                                     const ALPHA_INT *csr_row_ptr_D,
                                     const ALPHA_INT *csr_col_ind_D,
                                     ALPHA_Complex8 *csr_val_C,
                                     const ALPHA_INT *csr_row_ptr_C,
                                     ALPHA_INT *csr_col_ind_C,
                                     const alphasparse_dcu_mat_info_t info_C,
                                     void *temp_buffer);

alphasparseStatus_t dcu_spgemm_c_csr_transA(alphasparseHandle_t handle,
                                            ALPHA_INT m,
                                            ALPHA_INT n,
                                            ALPHA_INT k,
                                            const ALPHA_Complex8 alpha,
                                            ALPHA_INT nnz_A,
                                            const ALPHA_Complex8 *csr_val_A,
                                            const ALPHA_INT *csr_row_ptr_A,
                                            const ALPHA_INT *csr_col_ind_A,
                                            ALPHA_INT nnz_B,
                                            const ALPHA_Complex8 *csr_val_B,
                                            const ALPHA_INT *csr_row_ptr_B,
                                            const ALPHA_INT *csr_col_ind_B,
                                            const ALPHA_Complex8 beta,
                                            ALPHA_INT nnz_D,
                                            const ALPHA_Complex8 *csr_val_D,
                                            const ALPHA_INT *csr_row_ptr_D,
                                            const ALPHA_INT *csr_col_ind_D,
                                            ALPHA_Complex8 *csr_val_C,
                                            const ALPHA_INT *csr_row_ptr_C,
                                            ALPHA_INT *csr_col_ind_C,
                                            const alphasparse_dcu_mat_info_t info_C,
                                            void *temp_buffer);

alphasparseStatus_t dcu_spgemm_c_csr_transB(alphasparseHandle_t handle,
                                            ALPHA_INT m,
                                            ALPHA_INT n,
                                            ALPHA_INT k,
                                            const ALPHA_Complex8 alpha,
                                            ALPHA_INT nnz_A,
                                            const ALPHA_Complex8 *csr_val_A,
                                            const ALPHA_INT *csr_row_ptr_A,
                                            const ALPHA_INT *csr_col_ind_A,
                                            ALPHA_INT nnz_B,
                                            const ALPHA_Complex8 *csr_val_B,
                                            const ALPHA_INT *csr_row_ptr_B,
                                            const ALPHA_INT *csr_col_ind_B,
                                            const ALPHA_Complex8 beta,
                                            ALPHA_INT nnz_D,
                                            const ALPHA_Complex8 *csr_val_D,
                                            const ALPHA_INT *csr_row_ptr_D,
                                            const ALPHA_INT *csr_col_ind_D,
                                            ALPHA_Complex8 *csr_val_C,
                                            const ALPHA_INT *csr_row_ptr_C,
                                            ALPHA_INT *csr_col_ind_C,
                                            const alphasparse_dcu_mat_info_t info_C,
                                            void *temp_buffer);

alphasparseStatus_t dcu_spgemm_c_csr_transAB(alphasparseHandle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             const ALPHA_Complex8 alpha,
                                             ALPHA_INT nnz_A,
                                             const ALPHA_Complex8 *csr_val_A,
                                             const ALPHA_INT *csr_row_ptr_A,
                                             const ALPHA_INT *csr_col_ind_A,
                                             ALPHA_INT nnz_B,
                                             const ALPHA_Complex8 *csr_val_B,
                                             const ALPHA_INT *csr_row_ptr_B,
                                             const ALPHA_INT *csr_col_ind_B,
                                             const ALPHA_Complex8 beta,
                                             ALPHA_INT nnz_D,
                                             const ALPHA_Complex8 *csr_val_D,
                                             const ALPHA_INT *csr_row_ptr_D,
                                             const ALPHA_INT *csr_col_ind_D,
                                             ALPHA_Complex8 *csr_val_C,
                                             const ALPHA_INT *csr_row_ptr_C,
                                             ALPHA_INT *csr_col_ind_C,
                                             const alphasparse_dcu_mat_info_t info_C,
                                             void *temp_buffer);
// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparseStatus_t dcu_trsv_c_csr_n_lo(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const ALPHA_Complex8 *x,
                                        ALPHA_Complex8 *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*x
alphasparseStatus_t dcu_trsv_c_csr_u_lo(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const ALPHA_Complex8 *x,
                                        ALPHA_Complex8 *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*x
alphasparseStatus_t dcu_trsv_c_csr_n_hi(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const ALPHA_Complex8 *x,
                                        ALPHA_Complex8 *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*x
alphasparseStatus_t dcu_trsv_c_csr_u_hi(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const ALPHA_Complex8 *x,
                                        ALPHA_Complex8 *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_c_csr_n_lo_trans(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const ALPHA_Complex8 alpha,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const ALPHA_Complex8 *x,
                                              ALPHA_Complex8 *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_c_csr_u_lo_trans(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const ALPHA_Complex8 alpha,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const ALPHA_Complex8 *x,
                                              ALPHA_Complex8 *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_c_csr_n_hi_trans(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const ALPHA_Complex8 alpha,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const ALPHA_Complex8 *x,
                                              ALPHA_Complex8 *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_c_csr_u_hi_trans(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const ALPHA_Complex8 alpha,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const ALPHA_Complex8 *x,
                                              ALPHA_Complex8 *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_c_csr_n_lo_conj(alphasparseHandle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const ALPHA_Complex8 *x,
                                             ALPHA_Complex8 *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);
// alpha*inv(L^T)*x
alphasparseStatus_t dcu_trsv_c_csr_u_lo_conj(alphasparseHandle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const ALPHA_Complex8 *x,
                                             ALPHA_Complex8 *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_c_csr_n_hi_conj(alphasparseHandle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const ALPHA_Complex8 *x,
                                             ALPHA_Complex8 *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);
// alpha*inv(U^T)*x
alphasparseStatus_t dcu_trsv_c_csr_u_hi_conj(alphasparseHandle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const ALPHA_Complex8 *x,
                                             ALPHA_Complex8 *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsv_c_csr_n(alphasparseHandle_t handle,
                                       ALPHA_INT m,
                                       ALPHA_INT nnz,
                                       const ALPHA_Complex8 alpha,
                                       const ALPHA_Complex8 *csr_val,
                                       const ALPHA_INT *csr_row_ptr,
                                       const ALPHA_INT *csr_col_ind,
                                       alphasparse_dcu_mat_info_t info,
                                       const ALPHA_Complex8 *x,
                                       ALPHA_Complex8 *y,
                                       alphasparse_dcu_solve_policy_t policy,
                                       void *temp_buffer);
// alpha*x
alphasparseStatus_t dcu_diagsv_c_csr_u(alphasparseHandle_t handle,
                                       ALPHA_INT m,
                                       ALPHA_INT nnz,
                                       const ALPHA_Complex8 alpha,
                                       const ALPHA_Complex8 *csr_val,
                                       const ALPHA_INT *csr_row_ptr,
                                       const ALPHA_INT *csr_col_ind,
                                       alphasparse_dcu_mat_info_t info,
                                       const ALPHA_Complex8 *x,
                                       ALPHA_Complex8 *y,
                                       alphasparse_dcu_solve_policy_t policy,
                                       void *temp_buffer);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi(alphasparseHandle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const ALPHA_Complex8 alpha,
                                        const ALPHA_Complex8 *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        ALPHA_Complex8 *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi_transA(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi_transB(alphasparseHandle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const ALPHA_Complex8 alpha,
                                               const ALPHA_Complex8 *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);

// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_n_lo_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(L)*B
alphasparseStatus_t dcu_trsm_c_csr_u_lo_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_n_hi_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparseStatus_t dcu_trsm_c_csr_u_hi_transAB(alphasparseHandle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsm_c_csr_n_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t dcu_diagsm_c_csr_u_row(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparseStatus_t dcu_diagsm_c_csr_n_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparseStatus_t dcu_diagsm_c_csr_u_col(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

alphasparseStatus_t dcu_geam_c_csr(alphasparseHandle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   const ALPHA_Complex8 alpha,
                                   ALPHA_INT nnz_A,
                                   const ALPHA_Complex8 *csr_val_A,
                                   const ALPHA_INT *csr_row_ptr_A,
                                   const ALPHA_INT *csr_col_ind_A,
                                   const ALPHA_Complex8 beta,
                                   ALPHA_INT nnz_B,
                                   const ALPHA_Complex8 *csr_val_B,
                                   const ALPHA_INT *csr_row_ptr_B,
                                   const ALPHA_INT *csr_col_ind_B,
                                   ALPHA_Complex8 *csr_val_C,
                                   const ALPHA_INT *csr_row_ptr_C,
                                   ALPHA_INT *csr_col_ind_C);
                                   
alphasparseStatus_t dcu_set_value_c_csr(spmat_csr_c_t *A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex8 value);