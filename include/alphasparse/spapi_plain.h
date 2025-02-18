/*******************************************************************************
* Copyright 2005-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/
#pragma once

#include "spdef.h"
#include "types.h"

/*
    Perform computations based on created matrix handle

    Level 2
*/
/*   Computes y = alpha * A * x + beta * y   */
alphasparseStatus_t alphasparse_s_mv_plain(const alphasparseOperation_t operation,
                                          const float alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const float *x,
                                          const float beta,
                                          float *y);

alphasparseStatus_t alphasparse_d_mv_plain(const alphasparseOperation_t operation,
                                          const double alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const double *x,
                                          const double beta,
                                          double *y);

alphasparseStatus_t alphasparse_c_mv_plain(const alphasparseOperation_t operation,
                                          const ALPHA_Complex8 alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const ALPHA_Complex8 *x,
                                          const ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_mv_plain(const alphasparseOperation_t operation,
                                          const ALPHA_Complex16 alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const ALPHA_Complex16 *x,
                                          const ALPHA_Complex16 beta,
                                          ALPHA_Complex16 *y);

/*    Computes y = alpha * A * x + beta * y  and d = <x, y> , the l2 inner product */
alphasparseStatus_t alphasparse_s_dotmv_plain(const alphasparseOperation_t transA,
                                             const float alpha,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                             const float *x,
                                             const float beta,
                                             float *y,
                                             float *d);

alphasparseStatus_t alphasparse_d_dotmv_plain(const alphasparseOperation_t transA,
                                             const double alpha,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                             const double *x,
                                             const double beta,
                                             double *y,
                                             double *d);

alphasparseStatus_t alphasparse_c_dotmv_plain(const alphasparseOperation_t transA,
                                             const ALPHA_Complex8 alpha,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                             const ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 beta,
                                             ALPHA_Complex8 *y,
                                             ALPHA_Complex8 *d);

alphasparseStatus_t alphasparse_z_dotmv_plain(const alphasparseOperation_t transA,
                                             const ALPHA_Complex16 alpha,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                             const ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 beta,
                                             ALPHA_Complex16 *y,
                                             ALPHA_Complex16 *d);

/*   Solves triangular system y = alpha * A^{-1} * x   */
alphasparseStatus_t alphasparse_s_trsv_plain(const alphasparseOperation_t operation,
                                            const float alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const float *x,
                                            float *y);

alphasparseStatus_t alphasparse_d_trsv_plain(const alphasparseOperation_t operation,
                                            const double alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const double *x,
                                            double *y);

alphasparseStatus_t alphasparse_c_trsv_plain(const alphasparseOperation_t operation,
                                            const ALPHA_Complex8 alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const ALPHA_Complex8 *x,
                                            ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_trsv_plain(const alphasparseOperation_t operation,
                                            const ALPHA_Complex16 alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const ALPHA_Complex16 *x,
                                            ALPHA_Complex16 *y);

/*   Applies symmetric Gauss-Seidel preconditioner to symmetric system A * x = b, */
/*   that is, it solves:                                                          */
/*      x0       = alpha*x                                                        */
/*      (L+D)*x1 = b - U*x0                                                       */
/*      (D+U)*x  = b - L*x1                                                       */
/*                                                                                */
/*   SYMGS_MV also returns y = A*x                                                */
alphasparseStatus_t alphasparse_s_symgs_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const float alpha,
                                             const float *b,
                                             float *x);

alphasparseStatus_t alphasparse_d_symgs_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const double alpha,
                                             const double *b,
                                             double *x);

alphasparseStatus_t alphasparse_c_symgs_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 *b,
                                             ALPHA_Complex8 *x);

alphasparseStatus_t alphasparse_z_symgs_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const ALPHA_Complex16 alpha,
                                             const ALPHA_Complex16 *b,
                                             ALPHA_Complex16 *x);

alphasparseStatus_t alphasparse_s_symgs_mv_plain(const alphasparseOperation_t op,
                                                const alphasparse_matrix_t A,
                                                const struct alpha_matrix_descr descr,
                                                const float alpha,
                                                const float *b,
                                                float *x,
                                                float *y);

alphasparseStatus_t alphasparse_d_symgs_mv_plain(const alphasparseOperation_t op,
                                                const alphasparse_matrix_t A,
                                                const struct alpha_matrix_descr descr,
                                                const double alpha,
                                                const double *b,
                                                double *x,
                                                double *y);

alphasparseStatus_t alphasparse_c_symgs_mv_plain(const alphasparseOperation_t op,
                                                const alphasparse_matrix_t A,
                                                const struct alpha_matrix_descr descr,
                                                const ALPHA_Complex8 alpha,
                                                const ALPHA_Complex8 *b,
                                                ALPHA_Complex8 *x,
                                                ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_symgs_mv_plain(const alphasparseOperation_t op,
                                                const alphasparse_matrix_t A,
                                                const struct alpha_matrix_descr descr,
                                                const ALPHA_Complex16 alpha,
                                                const ALPHA_Complex16 *b,
                                                ALPHA_Complex16 *x,
                                                ALPHA_Complex16 *y);

/*   Computes an action of a preconditioner
         which corresponds to the approximate matrix decomposition A ≈ (L+D)*E*(U+D)
         for the system Ax = b.

         L is lower triangular part of A
         U is upper triangular part of A
         D is diagonal values of A 
         E is approximate diagonal inverse            
                                                                
         That is, it solves:                                      
             r = rhs - A*x0                                       
             (L + D)*E*(U + D)*dx = r                             
             x1 = x0 + dx                                        */

alphasparseStatus_t alphasparse_s_lu_smoother_plain(const alphasparseOperation_t op,
                                                   const alphasparse_matrix_t A,
                                                   const struct alpha_matrix_descr descr,
                                                   const float *diag,
                                                   const float *approx_diag_inverse,
                                                   float *x,
                                                   const float *rhs);

alphasparseStatus_t alphasparse_d_lu_smoother_plain(const alphasparseOperation_t op,
                                                   const alphasparse_matrix_t A,
                                                   const struct alpha_matrix_descr descr,
                                                   const double *diag,
                                                   const double *approx_diag_inverse,
                                                   double *x,
                                                   const double *rhs);

alphasparseStatus_t alphasparse_c_lu_smoother_plain(const alphasparseOperation_t op,
                                                   const alphasparse_matrix_t A,
                                                   const struct alpha_matrix_descr descr,
                                                   const ALPHA_Complex8 *diag,
                                                   const ALPHA_Complex8 *approx_diag_inverse,
                                                   ALPHA_Complex8 *x,
                                                   const ALPHA_Complex8 *rhs);

alphasparseStatus_t alphasparse_z_lu_smoother_plain(const alphasparseOperation_t op,
                                                   const alphasparse_matrix_t A,
                                                   const struct alpha_matrix_descr descr,
                                                   const ALPHA_Complex16 *diag,
                                                   const ALPHA_Complex16 *approx_diag_inverse,
                                                   ALPHA_Complex16 *x,
                                                   const ALPHA_Complex16 *rhs);

/* Level 3 */

/*   Computes y = alpha * A * x + beta * y   */
alphasparseStatus_t alphasparse_s_mm_plain(const alphasparseOperation_t operation,
                                          const float alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                          const float *x,
                                          const ALPHA_INT columns,
                                          const ALPHA_INT ldx,
                                          const float beta,
                                          float *y,
                                          const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_d_mm_plain(const alphasparseOperation_t operation,
                                          const double alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                          const double *x,
                                          const ALPHA_INT columns,
                                          const ALPHA_INT ldx,
                                          const double beta,
                                          double *y,
                                          const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_c_mm_plain(const alphasparseOperation_t operation,
                                          const ALPHA_Complex8 alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                          const ALPHA_Complex8 *x,
                                          const ALPHA_INT columns,
                                          const ALPHA_INT ldx,
                                          const ALPHA_Complex8 beta,
                                          ALPHA_Complex8 *y,
                                          const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_z_mm_plain(const alphasparseOperation_t operation,
                                          const ALPHA_Complex16 alpha,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                          const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                          const ALPHA_Complex16 *x,
                                          const ALPHA_INT columns,
                                          const ALPHA_INT ldx,
                                          const ALPHA_Complex16 beta,
                                          ALPHA_Complex16 *y,
                                          const ALPHA_INT ldy);

/*   Solves triangular system y = alpha * A^{-1} * x   */
alphasparseStatus_t alphasparse_s_trsm_plain(const alphasparseOperation_t operation,
                                            const float alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                            const float *x,
                                            const ALPHA_INT columns,
                                            const ALPHA_INT ldx,
                                            float *y,
                                            const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_d_trsm_plain(const alphasparseOperation_t operation,
                                            const double alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                            const double *x,
                                            const ALPHA_INT columns,
                                            const ALPHA_INT ldx,
                                            double *y,
                                            const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_c_trsm_plain(const alphasparseOperation_t operation,
                                            const ALPHA_Complex8 alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                            const ALPHA_Complex8 *x,
                                            const ALPHA_INT columns,
                                            const ALPHA_INT ldx,
                                            ALPHA_Complex8 *y,
                                            const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_z_trsm_plain(const alphasparseOperation_t operation,
                                            const ALPHA_Complex16 alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                            const ALPHA_Complex16 *x,
                                            const ALPHA_INT columns,
                                            const ALPHA_INT ldx,
                                            ALPHA_Complex16 *y,
                                            const ALPHA_INT ldy);

/* Sparse-sparse functionality */

/*   Computes sum of sparse matrices: C = alpha * op(A) + B, result is sparse   */
alphasparseStatus_t alphasparse_s_add_plain(const alphasparseOperation_t operation,
                                           const alphasparse_matrix_t A,
                                           const float alpha,
                                           const alphasparse_matrix_t B,
                                           alphasparse_matrix_t *C);

alphasparseStatus_t alphasparse_d_add_plain(const alphasparseOperation_t operation,
                                           const alphasparse_matrix_t A,
                                           const double alpha,
                                           const alphasparse_matrix_t B,
                                           alphasparse_matrix_t *C);

alphasparseStatus_t alphasparse_c_add_plain(const alphasparseOperation_t operation,
                                           const alphasparse_matrix_t A,
                                           const ALPHA_Complex8 alpha,
                                           const alphasparse_matrix_t B,
                                           alphasparse_matrix_t *C);

alphasparseStatus_t alphasparse_z_add_plain(const alphasparseOperation_t operation,
                                           const alphasparse_matrix_t A,
                                           const ALPHA_Complex16 alpha,
                                           const alphasparse_matrix_t B,
                                           alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = op(A) * B, result is sparse   */
alphasparseStatus_t alphasparse_spmm_plain(const alphasparseOperation_t operation,
                                          const alphasparse_matrix_t A,
                                          const alphasparse_matrix_t B,
                                          alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse   */
alphasparseStatus_t alphasparse_sp2m_plain(const alphasparseOperation_t transA,
                                          const struct alpha_matrix_descr descrA,
                                          const alphasparse_matrix_t A,
                                          const alphasparseOperation_t transB,
                                          const struct alpha_matrix_descr descrB,
                                          const alphasparse_matrix_t B,
                                          const alphasparse_request_t request,
                                          alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is sparse   */
alphasparseStatus_t alphasparse_syrk_plain(const alphasparseOperation_t operation,
                                          const alphasparse_matrix_t A,
                                          alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is sparse   */
alphasparseStatus_t alphasparse_sypr_plain(const alphasparseOperation_t transA,
                                          const alphasparse_matrix_t A,
                                          const alphasparse_matrix_t B,
                                          const struct alpha_matrix_descr descrB,
                                          alphasparse_matrix_t *C,
                                          const alphasparse_request_t request);

/*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is dense */
alphasparseStatus_t alphasparse_s_syprd_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const float *B,
                                             const alphasparse_layout_t layoutB,
                                             const ALPHA_INT ldb,
                                             const float alpha,
                                             const float beta,
                                             float *C,
                                             const alphasparse_layout_t layoutC,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_syprd_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const double *B,
                                             const alphasparse_layout_t layoutB,
                                             const ALPHA_INT ldb,
                                             const double alpha,
                                             const double beta,
                                             double *C,
                                             const alphasparse_layout_t layoutC,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_syprd_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const ALPHA_Complex8 *B,
                                             const alphasparse_layout_t layoutB,
                                             const ALPHA_INT ldb,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 beta,
                                             ALPHA_Complex8 *C,
                                             const alphasparse_layout_t layoutC,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_syprd_plain(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const ALPHA_Complex16 *B,
                                             const alphasparse_layout_t layoutB,
                                             const ALPHA_INT ldb,
                                             const ALPHA_Complex16 alpha,
                                             const ALPHA_Complex16 beta,
                                             ALPHA_Complex16 *C,
                                             const alphasparse_layout_t layoutC,
                                             const ALPHA_INT ldc);

/*   Computes product of sparse matrices: C = op(A) * B, result is dense   */
alphasparseStatus_t alphasparse_s_spmmd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const alphasparse_matrix_t B,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             float *C,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_spmmd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const alphasparse_matrix_t B,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             double *C,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_spmmd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const alphasparse_matrix_t B,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             ALPHA_Complex8 *C,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_spmmd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const alphasparse_matrix_t B,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             ALPHA_Complex16 *C,
                                             const ALPHA_INT ldc);

/*   Computes product of sparse matrices: C = opA(A) * opB(B), result is dense*/
alphasparseStatus_t alphasparse_s_sp2md_plain(const alphasparseOperation_t transA,
                                             const struct alpha_matrix_descr descrA,
                                             const alphasparse_matrix_t A,
                                             const alphasparseOperation_t transB,
                                             const struct alpha_matrix_descr descrB,
                                             const alphasparse_matrix_t B,
                                             const float alpha,
                                             const float beta,
                                             float *C,
                                             const alphasparse_layout_t layout,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_sp2md_plain(const alphasparseOperation_t transA,
                                             const struct alpha_matrix_descr descrA,
                                             const alphasparse_matrix_t A,
                                             const alphasparseOperation_t transB,
                                             const struct alpha_matrix_descr descrB,
                                             const alphasparse_matrix_t B,
                                             const double alpha,
                                             const double beta,
                                             double *C,
                                             const alphasparse_layout_t layout,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_sp2md_plain(const alphasparseOperation_t transA,
                                             const struct alpha_matrix_descr descrA,
                                             const alphasparse_matrix_t A,
                                             const alphasparseOperation_t transB,
                                             const struct alpha_matrix_descr descrB,
                                             const alphasparse_matrix_t B,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 beta,
                                             ALPHA_Complex8 *C,
                                             const alphasparse_layout_t layout,
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_sp2md_plain(const alphasparseOperation_t transA,
                                             const struct alpha_matrix_descr descrA,
                                             const alphasparse_matrix_t A,
                                             const alphasparseOperation_t transB,
                                             const struct alpha_matrix_descr descrB,
                                             const alphasparse_matrix_t B,
                                             const ALPHA_Complex16 alpha,
                                             const ALPHA_Complex16 beta,
                                             ALPHA_Complex16 *C,
                                             const alphasparse_layout_t layout,
                                             const ALPHA_INT ldc);

/*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is dense */
alphasparseStatus_t alphasparse_s_syrkd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const float alpha,
                                             const float beta,
                                             float *C,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_syrkd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const double alpha,
                                             const double beta,
                                             double *C,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_syrkd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 beta,
                                             ALPHA_Complex8 *C,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_syrkd_plain(const alphasparseOperation_t operation,
                                             const alphasparse_matrix_t A,
                                             const ALPHA_Complex16 alpha,
                                             const ALPHA_Complex16 beta,
                                             ALPHA_Complex16 *C,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_s_axpy_plain(const ALPHA_INT nz,
                                            const float a,
                                            const float *x,
                                            const ALPHA_INT *indx,
                                            float *y);

alphasparseStatus_t alphasparse_d_axpy_plain(const ALPHA_INT nz,
                                            const double a,
                                            const double *x,
                                            const ALPHA_INT *indx,
                                            double *y);

alphasparseStatus_t alphasparse_c_axpy_plain(const ALPHA_INT nz,
                                            const ALPHA_Complex8 a,
                                            const ALPHA_Complex8 *x,
                                            const ALPHA_INT *indx,
                                            ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_axpy_plain(const ALPHA_INT nz,
                                            const ALPHA_Complex16 a,
                                            const ALPHA_Complex16 *x,
                                            const ALPHA_INT *indx,
                                            ALPHA_Complex16 *y);

alphasparseStatus_t alphasparse_s_gthr_plain(const ALPHA_INT nz,
                                            const float *y,
                                            float *x,
                                            const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_d_gthr_plain(const ALPHA_INT nz,
                                            const double *y,
                                            double *x,
                                            const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_c_gthr_plain(const ALPHA_INT nz,
                                            const ALPHA_Complex8 *y,
                                            ALPHA_Complex8 *x,
                                            const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_z_gthr_plain(const ALPHA_INT nz,
                                            const ALPHA_Complex16 *y,
                                            ALPHA_Complex16 *x,
                                            const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_s_gthrz_plain(const ALPHA_INT nz,
                                             float *y,
                                             float *x,
                                             const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_d_gthrz_plain(const ALPHA_INT nz,
                                             double *y,
                                             double *x,
                                             const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_c_gthrz_plain(const ALPHA_INT nz,
                                             ALPHA_Complex8 *y,
                                             ALPHA_Complex8 *x,
                                             const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_z_gthrz_plain(const ALPHA_INT nz,
                                             ALPHA_Complex16 *y,
                                             ALPHA_Complex16 *x,
                                             const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_s_rot_plain(const ALPHA_INT nz,
                                           float *x,
                                           const ALPHA_INT *indx,
                                           float *y,
                                           const float c,
                                           const float s);

alphasparseStatus_t alphasparse_d_rot_plain(const ALPHA_INT nz,
                                           double *x,
                                           const ALPHA_INT *indx,
                                           double *y,
                                           const double c,
                                           const double s);

alphasparseStatus_t alphasparse_s_sctr_plain(const ALPHA_INT nz,
                                            const float *x,
                                            const ALPHA_INT *indx,
                                            float *y);

alphasparseStatus_t alphasparse_d_sctr_plain(const ALPHA_INT nz,
                                            const double *x,
                                            const ALPHA_INT *indx,
                                            double *y);

alphasparseStatus_t alphasparse_c_sctr_plain(const ALPHA_INT nz,
                                            const ALPHA_Complex8 *x,
                                            const ALPHA_INT *indx,
                                            ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_sctr_plain(const ALPHA_INT nz,
                                            const ALPHA_Complex16 *x,
                                            const ALPHA_INT *indx,
                                            ALPHA_Complex16 *y);

float alphasparse_s_doti_plain(const ALPHA_INT nz,
                              const float *x,
                              const ALPHA_INT *indx,
                              const float *y);

double alphasparse_d_doti_plain(const ALPHA_INT nz,
                               const double *x,
                               const ALPHA_INT *indx,
                               const double *y);

void alphasparse_c_dotci_sub_plain(const ALPHA_INT nz,
                                  const ALPHA_Complex8 *x,
                                  const ALPHA_INT *indx,
                                  const ALPHA_Complex8 *y,
                                  ALPHA_Complex8 *dutci);

void alphasparse_z_dotci_sub_plain(const ALPHA_INT nz,
                                  const ALPHA_Complex16 *x,
                                  const ALPHA_INT *indx,
                                  const ALPHA_Complex16 *y,
                                  ALPHA_Complex16 *dutci);

void alphasparse_c_dotui_sub_plain(const ALPHA_INT nz,
                                  const ALPHA_Complex8 *x,
                                  const ALPHA_INT *indx,
                                  const ALPHA_Complex8 *y,
                                  ALPHA_Complex8 *dutui);

void alphasparse_z_dotui_sub_plain(const ALPHA_INT nz,
                                  const ALPHA_Complex16 *x,
                                  const ALPHA_INT *indx,
                                  const ALPHA_Complex16 *y,
                                  ALPHA_Complex16 *dutui);

alphasparseStatus_t alphasparse_s_set_value_plain(alphasparse_matrix_t A,
                                                 const ALPHA_INT row,
                                                 const ALPHA_INT col,
                                                 const float value);

alphasparseStatus_t alphasparse_d_set_value_plain(alphasparse_matrix_t A,
                                                 const ALPHA_INT row,
                                                 const ALPHA_INT col,
                                                 const double value);

alphasparseStatus_t alphasparse_c_set_value_plain(alphasparse_matrix_t A,
                                                 const ALPHA_INT row,
                                                 const ALPHA_INT col,
                                                 const ALPHA_Complex8 value);

alphasparseStatus_t alphasparse_z_set_value_plain(alphasparse_matrix_t A,
                                                 const ALPHA_INT row,
                                                 const ALPHA_INT col,
                                                 const ALPHA_Complex16 value);

alphasparseStatus_t alphasparse_s_update_values_plain(alphasparse_matrix_t A,
                                                     const ALPHA_INT nvalues,
                                                     const ALPHA_INT *indx,
                                                     const ALPHA_INT *indy,
                                                     float *values);

alphasparseStatus_t alphasparse_d_update_values_plain(alphasparse_matrix_t A,
                                                     const ALPHA_INT nvalues,
                                                     const ALPHA_INT *indx,
                                                     const ALPHA_INT *indy,
                                                     double *values);

alphasparseStatus_t alphasparse_c_update_values_plain(alphasparse_matrix_t A,
                                                     const ALPHA_INT nvalues,
                                                     const ALPHA_INT *indx,
                                                     const ALPHA_INT *indy,
                                                     ALPHA_Complex8 *values);

alphasparseStatus_t alphasparse_z_update_values_plain(alphasparse_matrix_t A,
                                                     const ALPHA_INT nvalues,
                                                     const ALPHA_INT *indx,
                                                     const ALPHA_INT *indy,
                                                     ALPHA_Complex16 *values);
