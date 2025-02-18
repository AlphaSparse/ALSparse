#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_c_doti(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex8 *x_val,
                               const ALPHA_INT *x_ind,
                               const ALPHA_Complex8 *y,
                               ALPHA_Complex8 *result);

alphasparseStatus_t dcu_c_dotci(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex8 *x_val,
                                const ALPHA_INT *x_ind,
                                const ALPHA_Complex8 *y,
                                ALPHA_Complex8 *result);

alphasparseStatus_t dcu_c_axpyi(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex8 alpha,
                                const ALPHA_Complex8 *x_val,
                                const ALPHA_INT *x_ind,
                                ALPHA_Complex8 *y);

alphasparseStatus_t dcu_c_gthr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex8 *y,
                               ALPHA_Complex8 *x_val,
                               const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_c_gthrz(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex8 *y,
                                ALPHA_Complex8 *x_val,
                                const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_c_sctr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex8 *x_val,
                               const ALPHA_INT *x_ind,
                               ALPHA_Complex8 *y);

alphasparseStatus_t dcu_c_axpby(alphasparseHandle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);