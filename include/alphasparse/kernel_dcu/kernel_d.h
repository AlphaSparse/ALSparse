#pragma once

#include "../spmat.h"

alphasparseStatus_t dcu_d_doti(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const double *x_val,
                               const ALPHA_INT *x_ind,
                               const double *y,
                               double *result);

alphasparseStatus_t dcu_d_axpyi(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const double alpha,
                                const double *x_val,
                                const ALPHA_INT *x_ind,
                                double *y);

alphasparseStatus_t dcu_d_gthr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const double *y,
                               double *x_val,
                               const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_d_gthrz(alphasparseHandle_t handle,
                                ALPHA_INT nnz,
                                const double *y,
                                double *x_val,
                                const ALPHA_INT *x_ind);

alphasparseStatus_t dcu_d_roti(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               double *x_val,
                               const ALPHA_INT *x_ind,
                               double *y,
                               const double *c,
                               const double *s);

alphasparseStatus_t dcu_d_sctr(alphasparseHandle_t handle,
                               ALPHA_INT nnz,
                               const double *x_val,
                               const ALPHA_INT *x_ind,
                               double *y);

alphasparseStatus_t dcu_d_axpby(alphasparseHandle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);