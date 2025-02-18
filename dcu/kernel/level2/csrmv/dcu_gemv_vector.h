#pragma once
#ifdef __DCU__
#include <hip/hip_runtime.h>
#endif

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

template <ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE, typename TYPE>
__launch_bounds__(BLOCKSIZE)
    __global__ static void csr_gemv_vector_memalign(ALPHA_INT m,
                                                    TYPE alpha,
                                                    const ALPHA_INT *row_offset,
                                                    const ALPHA_INT *csr_col_ind,
                                                    const TYPE *csr_val,
                                                    const TYPE *x,
                                                    TYPE beta,
                                                    TYPE *y,
                                                    u_int32_t flag)
{
    const ALPHA_INT lid               = threadIdx.x & (WF_SIZE - 1); // thread index within the wavefront
    const ALPHA_INT VECTORS_PER_BLOCK = BLOCKSIZE / WF_SIZE; // vector num in the block
    const ALPHA_INT vector_lane       = threadIdx.x / WF_SIZE; // vector index within the block

    ALPHA_INT gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    ALPHA_INT nwf = gridDim.x * BLOCKSIZE / WF_SIZE;

    // Loop over rows
    for (ALPHA_INT row = gid / WF_SIZE; row < m; row += nwf) {
        ALPHA_INT row_start, row_end;
        row_start = row_offset[row];
        row_end   = row_offset[row + 1];

        TYPE sum = TYPE{};
        if (WF_SIZE == 16 && row_end - row_start > 16) {
            // ensure aligned memory access to csr_col_ind and csr_val
            ALPHA_INT j = row_start - (row_start & (WF_SIZE - 1)) + lid;

            // accumulate local sums
            if (j >= row_start && j < row_end) {
                // sum += csr_val[j] * x[csr_col_ind[j]];
                sum = alpha_madd(csr_val[j], x[csr_col_ind[j]], sum);
            }

            // accumulate local sums
            for (j += WF_SIZE; j < row_end; j += WF_SIZE) {
                // sum += csr_val[j] * x[csr_col_ind[j]];
                sum = alpha_madd(csr_val[j], x[csr_col_ind[j]], sum);
            }
        } else {
            // Loop over non-zero elements
            for (ALPHA_INT j = row_start + lid; j < row_end; j += WF_SIZE) {
                // sum += alpha * csr_val[j] * x[csr_col_ind[j]];
                sum = alpha_madd(csr_val[j], x[csr_col_ind[j]], sum);
            }
        }

        // Obtain row sum using parallel reduction
        sum = wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1) {
            TYPE t1, t2;
            t1 = alpha_mul(y[row], beta);
            t2 = alpha_mul(sum, alpha);
            y[row] = alpha_add(t1, t2);
        }
    }
}

template <typename TYPE, ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE>
__launch_bounds__(BLOCKSIZE)
    __global__ static void csr_gemv_vector(ALPHA_INT m,
                                           TYPE alpha,
                                           const ALPHA_INT *row_offset,
                                           const ALPHA_INT *csr_col_ind,
                                           const TYPE *csr_val,
                                           const TYPE *x,
                                           TYPE beta,
                                           TYPE *y,
                                           u_int32_t flag)
{
    const ALPHA_INT lid               = threadIdx.x & (WF_SIZE - 1); // thread index within the wavefront
    const ALPHA_INT VECTORS_PER_BLOCK = BLOCKSIZE / WF_SIZE; // vector num in the block
    const ALPHA_INT vector_lane       = threadIdx.x / WF_SIZE; // vector index within the block

    ALPHA_INT gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    ALPHA_INT nwf = gridDim.x * BLOCKSIZE / WF_SIZE;

    // Loop over rows
    for (ALPHA_INT row = gid / WF_SIZE; row < m; row += nwf) {
        ALPHA_INT row_start, row_end;

        row_start = row_offset[row];
        row_end   = row_offset[row + 1];

        TYPE sum;
        sum = alpha_setzero(sum);

        // Loop over non-zero elements
        for (ALPHA_INT j = row_start + lid; j < row_end; j += WF_SIZE) {
            // sum += csr_val[j] * x[csr_col_ind[j]];
            sum = alpha_madd(csr_val[j], x[csr_col_ind[j]], sum);
        }

        // Obtain row sum using parallel reduction
        sum = wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1) {
            TYPE t1, t2;
            t1 = alpha_mul(y[row], beta);
            t2 = alpha_mul(sum, alpha);
            y[row] = alpha_add(t1, t2);
        }
    }
}

#define CSRGEMV_VECTOR_ALIGN(WFSIZE)                                       \
    {                                                                      \
        dim3 csrmvn_blocks(block_num_base *WFSIZE);                        \
        dim3 csrmvn_threads(CSRMVN_DIM);                                   \
        hipLaunchKernelGGL((csr_gemv_vector_memalign<CSRMVN_DIM, WFSIZE>), \
                           csrmvn_blocks,                                  \
                           csrmvn_threads,                                 \
                           0,                                              \
                           handle->stream,                                 \
                           m,                                              \
                           alpha,                                          \
                           csr_row_ptr,                                    \
                           csr_col_ind,                                    \
                           csr_val,                                        \
                           x,                                              \
                           beta,                                           \
                           y,                                              \
                           flag);                                          \
    }

#define CSRGEMV_VECTOR_ALIGN_LDSREDUCE(WFSIZE)                                       \
    {                                                                                \
        dim3 csrmvn_blocks(block_num_base *WFSIZE);                                  \
        dim3 csrmvn_threads(CSRMVN_DIM);                                             \
        hipLaunchKernelGGL((csr_gemv_vector_memalign_ldsreduce<CSRMVN_DIM, WFSIZE>), \
                           csrmvn_blocks,                                            \
                           csrmvn_threads,                                           \
                           0,                                                        \
                           handle->stream,                                           \
                           m,                                                        \
                           alpha,                                                    \
                           csr_row_ptr,                                              \
                           csr_col_ind,                                              \
                           csr_val,                                                  \
                           x,                                                        \
                           beta,                                                     \
                           y);                                                       \
    }

template <typename TYPE>
alphasparseStatus_t csr_gemv_vector_dispatch(alphasparseHandle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const TYPE alpha,
                                              const TYPE *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const TYPE *x,
                                              const TYPE beta,
                                              TYPE *y,
                                              u_int32_t flag)
{
    const ALPHA_INT CSRMVN_DIM  = 512;
    const ALPHA_INT nnz_per_row = nnz / m;

    const ALPHA_INT block_num_base = (m - 1) / CSRMVN_DIM + 1;

    if (handle->check_flag) {
        if (nnz_per_row < 4) {
            CSRGEMV_VECTOR_ALIGN(2);
        } else if (nnz_per_row < 8) {
            CSRGEMV_VECTOR_ALIGN(4);
        } else if (nnz_per_row < 16) {
            CSRGEMV_VECTOR_ALIGN(8);
        } else if (nnz_per_row < 32) {
            CSRGEMV_VECTOR_ALIGN(16);
        } else if (nnz_per_row < 64) {
            CSRGEMV_VECTOR_ALIGN(32);
        } else {
            CSRGEMV_VECTOR_ALIGN(64);
        }

        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // if (nnz_per_row < 4) {
    //     CSRGEMV_VECTOR_ALIGN_LDSREDUCE(2);
    // } else if (nnz_per_row < 8) {
    //     CSRGEMV_VECTOR_ALIGN_LDSREDUCE(4);
    // } else if (nnz_per_row < 16) {
    //     CSRGEMV_VECTOR_ALIGN_LDSREDUCE(8);
    // } else if (nnz_per_row < 32) {
    //     CSRGEMV_VECTOR_ALIGN_LDSREDUCE(16);
    // } else if (nnz_per_row < 64) {
    //     CSRGEMV_VECTOR_ALIGN_LDSREDUCE(32);
    // } else {
    //     CSRGEMV_VECTOR_ALIGN_LDSREDUCE(64);
    // }

    if (nnz_per_row < 4) {
        CSRGEMV_VECTOR_ALIGN(2);
    } else if (nnz_per_row < 8) {
        CSRGEMV_VECTOR_ALIGN(4);
    } else if (nnz_per_row < 16) {
        CSRGEMV_VECTOR_ALIGN(8);
    } else if (nnz_per_row < 32) {
        CSRGEMV_VECTOR_ALIGN(16);
    } else if (nnz_per_row < 64) {
        CSRGEMV_VECTOR_ALIGN(32);
    } else {
        CSRGEMV_VECTOR_ALIGN(64);
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
