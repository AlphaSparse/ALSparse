#pragma once

#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

template <ALPHA_INT UNROLL>
__global__ static void
csr_gemv_scalar_unroll(ALPHA_INT m,
                       ALPHA_INT n,
                       ALPHA_INT nnz,
                       const ALPHA_Number alpha,
                       const ALPHA_Number *csr_val,
                       const ALPHA_INT *csr_row_ptr,
                       const ALPHA_INT *csr_col_ind,
                       const ALPHA_Number *x,
                       const ALPHA_Number beta,
                       ALPHA_Number *y,
                       u_int32_t flag)
{
    ALPHA_INT ix     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    for (ALPHA_INT i = ix; i < m; i += stride) {
        alpha_mule(y[i], beta);
        ALPHA_Number tmp;
        alpha_setzero(tmp);

        if (UNROLL == 2) {
            ALPHA_Number t1, t2;
            alpha_setzero(t1);
            alpha_setzero(t2);
            ALPHA_INT j = csr_row_ptr[i];
            for (; j < csr_row_ptr[i + 1] - 1; j += 2) {
                alpha_madde(t1, csr_val[j], x[csr_col_ind[j]]);
                alpha_madde(t2, csr_val[j + 1], x[csr_col_ind[j + 1]]);
            }
            for (; j < csr_row_ptr[i + 1]; ++j) {
                alpha_madde(tmp, csr_val[j], x[csr_col_ind[j]]);
            }
            alpha_add(tmp, tmp, t1);
            alpha_add(tmp, tmp, t2);
        } else if (UNROLL == 4) {
            ALPHA_Number t1, t2, t3, t4;
            alpha_setzero(t1);
            alpha_setzero(t2);
            alpha_setzero(t3);
            alpha_setzero(t4);
            ALPHA_INT j = csr_row_ptr[i];
            for (; j < csr_row_ptr[i + 1] - 3; j += 4) {
                alpha_madde(t1, csr_val[j], x[csr_col_ind[j]]);
                alpha_madde(t2, csr_val[j + 1], x[csr_col_ind[j + 1]]);
                alpha_madde(t3, csr_val[j + 2], x[csr_col_ind[j + 2]]);
                alpha_madde(t4, csr_val[j + 3], x[csr_col_ind[j + 3]]);
            }
            for (; j < csr_row_ptr[i + 1]; ++j) {
                alpha_madde(tmp, csr_val[j], x[csr_col_ind[j]]);
            }
            alpha_add(tmp, tmp, t1);
            alpha_add(tmp, tmp, t2);
            alpha_add(tmp, tmp, t3);
            alpha_add(tmp, tmp, t4);
        } else if (UNROLL == 8) {
            ALPHA_Number t1, t2, t3, t4, t5, t6, t7, t8;
            alpha_setzero(t1);
            alpha_setzero(t2);
            alpha_setzero(t3);
            alpha_setzero(t4);
            alpha_setzero(t5);
            alpha_setzero(t6);
            alpha_setzero(t7);
            alpha_setzero(t8);
            ALPHA_INT j = csr_row_ptr[i];
            for (; j < csr_row_ptr[i + 1] - 7; j += 8) {
                alpha_madde(t1, csr_val[j], x[csr_col_ind[j]]);
                alpha_madde(t2, csr_val[j + 1], x[csr_col_ind[j + 1]]);
                alpha_madde(t3, csr_val[j + 2], x[csr_col_ind[j + 2]]);
                alpha_madde(t4, csr_val[j + 3], x[csr_col_ind[j + 3]]);
                alpha_madde(t5, csr_val[j + 4], x[csr_col_ind[j + 4]]);
                alpha_madde(t6, csr_val[j + 5], x[csr_col_ind[j + 5]]);
                alpha_madde(t7, csr_val[j + 6], x[csr_col_ind[j + 6]]);
                alpha_madde(t8, csr_val[j + 7], x[csr_col_ind[j + 7]]);
            }
            for (; j < csr_row_ptr[i + 1]; ++j) {
                alpha_madde(tmp, csr_val[j], x[csr_col_ind[j]]);
            }
            alpha_add(tmp, tmp, t1);
            alpha_add(tmp, tmp, t2);
            alpha_add(tmp, tmp, t3);
            alpha_add(tmp, tmp, t4);
            alpha_add(tmp, tmp, t5);
            alpha_add(tmp, tmp, t6);
            alpha_add(tmp, tmp, t7);
            alpha_add(tmp, tmp, t8);
        } else {
#pragma unroll UNROLL
            for (ALPHA_INT j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++) {
                alpha_madde(tmp, csr_val[j], x[csr_col_ind[j]]);
            }
        }

        alpha_madde(y[i], alpha, tmp);
    }
}

__global__ static void
csr_gemv_scalar(ALPHA_INT m,
                ALPHA_INT n,
                ALPHA_INT nnz,
                const ALPHA_Number alpha,
                const ALPHA_Number *csr_val,
                const ALPHA_INT *csr_row_ptr,
                const ALPHA_INT *csr_col_ind,
                const ALPHA_Number *x,
                const ALPHA_Number beta,
                ALPHA_Number *y,
                u_int32_t flag)
{
    ALPHA_INT ix     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    for (ALPHA_INT i = ix; i < m; i += stride) {
        alpha_mule(y[i], beta);
        ALPHA_Number tmp;
        alpha_setzero(tmp);

        for (ALPHA_INT j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++) {
            alpha_madde(tmp, csr_val[j], x[csr_col_ind[j]]);
        }

        alpha_madde(y[i], alpha, tmp);
    }
}

alphasparse_status_t csr_gemv_scalar_dispatch(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const ALPHA_Number alpha,
                                              const ALPHA_Number *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y,
                                              u_int32_t flag)
{
    const ALPHA_INT threadPerBlock = 256;
    const ALPHA_INT blockPerGrid   = (m - 1) / threadPerBlock + 1;
    const ALPHA_INT UNROLL         = 1; //TODO how to tune this param
    // hipLaunchKernelGGL((csr_gemv_scalar_unroll<UNROLL>), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    hipLaunchKernelGGL((csr_gemv_scalar), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}