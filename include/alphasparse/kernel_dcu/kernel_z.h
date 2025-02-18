#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_z_doti(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex16 *x_val,
                               const ALPHA_INT *x_ind,
                               const ALPHA_Complex16 *y,
                               ALPHA_Complex16 *result);

alphasparseStatus_t dcu_z_dotci(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex16 *x_val,
                                const ALPHA_INT *x_ind,
                                const ALPHA_Complex16 *y,
                                ALPHA_Complex16 *result);

alphasparseStatus_t dcu_z_axpyi(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex16 alpha,
                                const ALPHA_Complex16 *x_val,
                                const ALPHA_INT *x_ind,
                                ALPHA_Complex16 *y);

alphasparseStatus_t dcu_z_gthr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex16 *y,
                               ALPHA_Complex16 *x_val,
                               const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_z_gthrz(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex16 *y,
                                ALPHA_Complex16 *x_val,
                                const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_z_sctr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex16 *x_val,
                               const ALPHA_INT *x_ind,
                               ALPHA_Complex16 *y);

alphasparseStatus_t dcu_z_axpby(alphasparseHandle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);