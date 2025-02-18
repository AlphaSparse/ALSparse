#pragma once

#include "../spmat.h"

// mv
alphasparseStatus_t dcu_gemv_s_csr5(alphasparseHandle_t handle,
                                     const float alpha,
                                     const spmat_csr5_s_t *csr5,
                                     alphasparse_dcu_mat_info_t info,
                                     const float *x,
                                     const float beta,
                                     float *y);

alphasparseStatus_t dcu_gemv_s_cooaos(alphasparseHandle_t handle,
                                       const float alpha,
                                       const spmat_cooaos_s_t *cooaos,
                                       alphasparse_dcu_mat_info_t info,
                                       const float *x,
                                       const float beta,
                                       float *y);

alphasparseStatus_t dcu_gemv_s_sell_csigma(alphasparseHandle_t handle,
                                            const float alpha,
                                            const spmat_sell_csigma_s_t *sell_csigma,
                                            alphasparse_dcu_mat_info_t info,
                                            const float *x,
                                            const float beta,
                                            float *y);

alphasparseStatus_t dcu_gemv_s_ellr(alphasparseHandle_t handle,
                                     const float alpha,
                                     const spmat_ellr_s_t *ellr,
                                     alphasparse_dcu_mat_info_t info,
                                     const float *x,
                                     const float beta,
                                     float *y);

alphasparseStatus_t dcu_gemv_s_csc(alphasparseHandle_t handle,
                                    const float alpha,
                                    const spmat_csc_s_t *csc,
                                    alphasparse_dcu_mat_info_t info,
                                    const float *x,
                                    const float beta,
                                    float *y);

alphasparseStatus_t dcu_gemv_s_dia(alphasparseHandle_t handle,
                                    const float alpha,
                                    const spmat_dia_s_t *dia,
                                    alphasparse_dcu_mat_info_t info,
                                    const float *x,
                                    const float beta,
                                    float *y);