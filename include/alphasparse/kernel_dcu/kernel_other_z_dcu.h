#pragma once

#include "../spmat.h"

// mv
alphasparseStatus_t dcu_gemv_z_csr5(alphasparseHandle_t handle,
                                     const ALPHA_Complex16 alpha,
                                     const spmat_csr5_z_t *csr5,
                                     alphasparse_dcu_mat_info_t info,
                                     const ALPHA_Complex16 *x,
                                     const ALPHA_Complex16 beta,
                                     ALPHA_Complex16 *y);

alphasparseStatus_t dcu_gemv_z_cooaos(alphasparseHandle_t handle,
                                       const ALPHA_Complex16 alpha,
                                       const spmat_cooaos_z_t *cooaos,
                                       alphasparse_dcu_mat_info_t info,
                                       const ALPHA_Complex16 *x,
                                       const ALPHA_Complex16 beta,
                                       ALPHA_Complex16 *y);

alphasparseStatus_t dcu_gemv_z_sell_csigma(alphasparseHandle_t handle,
                                            const ALPHA_Complex16 alpha,
                                            const spmat_sell_csigma_z_t *sell_csigma,
                                            alphasparse_dcu_mat_info_t info,
                                            const ALPHA_Complex16 *x,
                                            const ALPHA_Complex16 beta,
                                            ALPHA_Complex16 *y);

alphasparseStatus_t dcu_gemv_z_ellr(alphasparseHandle_t handle,
                                     const ALPHA_Complex16 alpha,
                                     const spmat_ellr_z_t *ellr,
                                     alphasparse_dcu_mat_info_t info,
                                     const ALPHA_Complex16 *x,
                                     const ALPHA_Complex16 beta,
                                     ALPHA_Complex16 *y);

alphasparseStatus_t dcu_gemv_z_csc(alphasparseHandle_t handle,
                                    const ALPHA_Complex16 alpha,
                                    const spmat_csc_z_t *csc,
                                    alphasparse_dcu_mat_info_t info,
                                    const ALPHA_Complex16 *x,
                                    const ALPHA_Complex16 beta,
                                    ALPHA_Complex16 *y);

alphasparseStatus_t dcu_gemv_z_dia(alphasparseHandle_t handle,
                                    const ALPHA_Complex16 alpha,
                                    const spmat_dia_z_t *dia,
                                    alphasparse_dcu_mat_info_t info,
                                    const ALPHA_Complex16 *x,
                                    const ALPHA_Complex16 beta,
                                    ALPHA_Complex16 *y);
