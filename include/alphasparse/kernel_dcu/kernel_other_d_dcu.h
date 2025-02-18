#pragma once

#include "../spmat.h"

// mv
alphasparseStatus_t dcu_gemv_d_csr5(alphasparseHandle_t handle,
                                     const double alpha,
                                     const spmat_csr5_d_t *csr5,
                                     alphasparse_dcu_mat_info_t info,
                                     const double *x,
                                     const double beta,
                                     double *y);

alphasparseStatus_t dcu_gemv_d_cooaos(alphasparseHandle_t handle,
                                       const double alpha,
                                       const spmat_cooaos_d_t *cooaos,
                                       alphasparse_dcu_mat_info_t info,
                                       const double *x,
                                       const double beta,
                                       double *y);

alphasparseStatus_t dcu_gemv_d_sell_csigma(alphasparseHandle_t handle,
                                            const double alpha,
                                            const spmat_sell_csigma_d_t *sell_csigma,
                                            alphasparse_dcu_mat_info_t info,
                                            const double *x,
                                            const double beta,
                                            double *y);

alphasparseStatus_t dcu_gemv_d_ellr(alphasparseHandle_t handle,
                                     const double alpha,
                                     const spmat_ellr_d_t *ellr,
                                     alphasparse_dcu_mat_info_t info,
                                     const double *x,
                                     const double beta,
                                     double *y);

alphasparseStatus_t dcu_gemv_d_csc(alphasparseHandle_t handle,
                                    const double alpha,
                                    const spmat_csc_d_t *csc,
                                    alphasparse_dcu_mat_info_t info,
                                    const double *x,
                                    const double beta,
                                    double *y);

alphasparseStatus_t dcu_gemv_d_dia(alphasparseHandle_t handle,
                                    const double alpha,
                                    const spmat_dia_d_t *dia,
                                    alphasparse_dcu_mat_info_t info,
                                    const double *x,
                                    const double beta,
                                    double *y);
