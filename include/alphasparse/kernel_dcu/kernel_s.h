#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_s_doti(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const float *x_val,
                               const ALPHA_INT *x_ind,
                               const float *y,
                               float *result);

alphasparseStatus_t dcu_s_axpyi(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const float alpha,
                                const float *x_val,
                                const ALPHA_INT *x_ind,
                                float *y);

alphasparseStatus_t dcu_s_gthr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const float *y,
                               float *x_val,
                               const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_s_gthrz(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const float *y,
                                float *x_val,
                                const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_s_roti(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               float *x_val,
                               const ALPHA_INT *x_ind,
                               float *y,
                               const float *c,
                               const float *s);

alphasparseStatus_t dcu_s_sctr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const float *x_val,
                               const ALPHA_INT *x_ind,
                               float *y);

alphasparseStatus_t dcu_s_axpby(alphasparseHandle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);