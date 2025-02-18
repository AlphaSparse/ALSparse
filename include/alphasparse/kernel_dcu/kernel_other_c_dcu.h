#pragma once

#include "../spmat.h"

// mv
alphasparseStatus_t dcu_gemv_c_csr5(alphasparseHandle_t handle,
                                     const ALPHA_Complex8 alpha,
                                     const spmat_csr5_c_t *csr5,
                                     alphasparse_dcu_mat_info_t info,
                                     const ALPHA_Complex8 *x,
                                     const ALPHA_Complex8 beta,
                                     ALPHA_Complex8 *y);

alphasparseStatus_t dcu_gemv_c_cooaos(alphasparseHandle_t handle,
                                       const ALPHA_Complex8 alpha,
                                       const spmat_cooaos_c_t *cooaos,
                                       alphasparse_dcu_mat_info_t info,
                                       const ALPHA_Complex8 *x,
                                       const ALPHA_Complex8 beta,
                                       ALPHA_Complex8 *y);

alphasparseStatus_t dcu_gemv_c_sell_csigma(alphasparseHandle_t handle,
                                            const ALPHA_Complex8 alpha,
                                            const spmat_sell_csigma_c_t *sell_csigma,
                                            alphasparse_dcu_mat_info_t info,
                                            const ALPHA_Complex8 *x,
                                            const ALPHA_Complex8 beta,
                                            ALPHA_Complex8 *y);

alphasparseStatus_t dcu_gemv_c_ellr(alphasparseHandle_t handle,
                                     const ALPHA_Complex8 alpha,
                                     const spmat_ellr_c_t *ellr,
                                     alphasparse_dcu_mat_info_t info,
                                     const ALPHA_Complex8 *x,
                                     const ALPHA_Complex8 beta,
                                     ALPHA_Complex8 *y);

alphasparseStatus_t dcu_gemv_c_csc(alphasparseHandle_t handle,
                                    const ALPHA_Complex8 alpha,
                                    const spmat_csc_c_t *csc,
                                    alphasparse_dcu_mat_info_t info,
                                    const ALPHA_Complex8 *x,
                                    const ALPHA_Complex8 beta,
                                    ALPHA_Complex8 *y);

alphasparseStatus_t dcu_gemv_c_dia(alphasparseHandle_t handle,
                                    const ALPHA_Complex8 alpha,
                                    const spmat_dia_c_t *dia,
                                    alphasparse_dcu_mat_info_t info,
                                    const ALPHA_Complex8 *x,
                                    const ALPHA_Complex8 beta,
                                    ALPHA_Complex8 *y);
