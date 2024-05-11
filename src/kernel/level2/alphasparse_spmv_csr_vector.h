#pragma once

#include "alphasparse.h"

template <int BLOCK_SIZE, int WF_SIZE, typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCK_SIZE)
    __global__ static void csr_gemv_vector_memalign(T m,
                                                    W alpha,
                                                    const T *csr_row_ptr,
                                                    const T *csr_col_ind,
                                                    const U *csr_val,
                                                    const U *x,
                                                    W beta,
                                                    V *y)
{
    const T lid = threadIdx.x & (WF_SIZE - 1);        // thread index within the wavefront
    const T VECTORS_PER_BLOCK = BLOCK_SIZE / WF_SIZE; // vector num in the block
    const T vector_lane = threadIdx.x / WF_SIZE;      // vector index within the block

    T gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    T nwf = gridDim.x * BLOCK_SIZE / WF_SIZE;
    // typedef cub::WarpReduce<ALPHA_Number, WF_SIZE> WarpReduce;
    // __shared__ typename WarpReduce::TempStorage temp_storage[VECTORS_PER_BLOCK];
    // Loop over rows
    for (T row = gid / WF_SIZE; row < m; row += nwf)
    {
        T row_start, row_end;
        row_start = csr_row_ptr[row];
        row_end = csr_row_ptr[row + 1];

        V sum = {};
        if (WF_SIZE >= 16 && row_end - row_start > 16)
        {
            // ensure aligned memory access to csr_col_ind and csr_val
            T j = row_start - (row_start & (WF_SIZE - 1)) + lid;

            // accumulate local sums
            if (j >= row_start && j < row_end)
            {
                sum += csr_val[j] * x[csr_col_ind[j]];
                // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
            }

            // accumulate local sums
            for (j += WF_SIZE; j < row_end; j += WF_SIZE)
            {
                sum += csr_val[j] * x[csr_col_ind[j]];
                // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
            }
        }
        else
        {
            // Loop over non-zero elements
            for (T j = row_start + lid; j < row_end; j += WF_SIZE)
            {
                sum += csr_val[j] * x[csr_col_ind[j]];
                // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
            }
        }
        // __syncthreads();
        // Obtain row sum using parallel reduction
        // sum = WarpReduce(temp_storage[vector_lane]).Sum(sum);
        sum = wfreduce_sum<WF_SIZE>(sum);
        // __syncthreads();
        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1)
        {
            y[row] = y[row] * beta + sum * alpha;
        }
    }
}

template <int BLOCK_SIZE, int WF_SIZE, typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCK_SIZE)
    __global__ static void csr_gemv_vector_memalign_ldsreduce(T m,
                                                              W alpha,
                                                              const T *row_offset,
                                                              const T *csr_col_ind,
                                                              const U *csr_val,
                                                              const U *x,
                                                              W beta,
                                                              V *y)
{
    const T tid = threadIdx.x;
    const T lid = threadIdx.x & (WF_SIZE - 1);        // thread index within the wavefront
    const T VECTORS_PER_BLOCK = BLOCK_SIZE / WF_SIZE; // vector num in the block
    const T vector_lane = threadIdx.x / WF_SIZE;      // vector index within the block

    T gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    T nwf = gridDim.x * BLOCK_SIZE / WF_SIZE;
    // typedef cub::WarpReduce<ALPHA_Number, WF_SIZE> WarpReduce;
    // __shared__ typename WarpReduce::TempStorage temp_storage[VECTORS_PER_BLOCK];
    // Loop over rows
    T row = gid / WF_SIZE;
    if (row < m)
    {
        T row_start, row_end;
        row_start = row_offset[row];
        row_end = row_offset[row + 1];
        V sum = V{};
        if (WF_SIZE >= 16 && row_end - row_start > WF_SIZE)
        {
            // ensure aligned memory access to csr_col_ind and csr_val
            T j = row_start - (row_start & (WF_SIZE - 1)) + lid;
            // accumulate local sums
            if (j >= row_start && j < row_end)
            {
                sum += csr_val[j] * x[csr_col_ind[j]];
                // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
            }
            // accumulate local sums
            for (j += WF_SIZE; j < row_end; j += WF_SIZE)
            {
                sum += csr_val[j] * x[csr_col_ind[j]];
                // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
            }
        }
        else
        {
            // Loop over non-zero elements
            for (T j = row_start + lid; j < row_end; j += WF_SIZE)
            {
                sum += csr_val[j] * x[csr_col_ind[j]];
                // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
            }
        }
        sum = wfreduce_sum<WF_SIZE>(sum);
        // __syncthreads();
        // Obtain row sum using parallel reduction
        // sum = WarpReduce(temp_storage[vector_lane]).Sum(sum);
        //  __syncthreads();
        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1)
        {
            y[row] = y[row] * beta + sum * alpha;
        }
    }
}

template <int BLOCK_SIZE, int WF_SIZE, typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCK_SIZE)
    __global__ static void csr_gemv_vector(T m,
                                           W alpha,
                                           const T *csr_row_ptr,
                                           const T *csr_col_ind,
                                           const U *csr_val,
                                           const U *x,
                                           W beta,
                                           V *y)
{
    const T lid = threadIdx.x & (WF_SIZE - 1);        // thread index within the wavefront
    const T VECTORS_PER_BLOCK = BLOCK_SIZE / WF_SIZE; // vector num in the block
    const T vector_lane = threadIdx.x / WF_SIZE;      // vector index within the block

    T gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    T nwf = gridDim.x * BLOCK_SIZE / WF_SIZE;

    // Loop over rows
    for (T row = gid / WF_SIZE; row < m; row += nwf)
    {
        T row_start, row_end;

        row_start = csr_row_ptr[row];
        row_end = csr_row_ptr[row + 1];

        V sum = {};

        // Loop over non-zero elements
        for (T j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
        }

        // Obtain row sum using parallel reduction
        sum = wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1)
        {
            y[row] = y[row] * beta + sum * alpha;
        }
    }
}

#define CSRGEMV_VECTOR_ALIGN(WFSIZE)                    \
    {                                                   \
        dim3 csrmvn_blocks(block_num_base *WFSIZE);     \
        dim3 csrmvn_threads(CSRMVN_DIM);                \
        csr_gemv_vector_memalign<CSRMVN_DIM, WFSIZE><<< \
            csrmvn_blocks,                              \
            csrmvn_threads,                             \
            0,                                          \
            handle->stream>>>(                          \
            m,                                          \
            alpha,                                      \
            csr_row_ptr,                                \
            csr_col_ind,                                \
            csr_val,                                    \
            x,                                          \
            beta,                                       \
            y);                                         \
    }

#define CSRGEMV_VECTOR_ALIGN_LDSREDUCE(WFSIZE)                    \
    {                                                             \
        dim3 csrmvn_blocks(block_num_base *WFSIZE);               \
        dim3 csrmvn_threads(CSRMVN_DIM);                          \
        csr_gemv_vector_memalign_ldsreduce<CSRMVN_DIM, WFSIZE><<< \
            csrmvn_blocks,                                        \
            csrmvn_threads,                                       \
            0,                                                    \
            handle->stream>>>(                                    \
            m,                                                    \
            alpha,                                                \
            csr_row_ptr,                                          \
            csr_col_ind,                                          \
            csr_val,                                              \
            x,                                                    \
            beta,                                                 \
            y);                                                   \
    }

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_vector(alphasparseHandle_t handle,
                                    T m,
                                    T n,
                                    T nnz,
                                    const W alpha,
                                    const U *csr_val,
                                    const T *csr_row_ptr,
                                    const T *csr_col_ind,
                                    const U *x,
                                    const W beta,
                                    V *y)
{
    const T CSRMVN_DIM = 512;
    const T nnz_per_row = nnz / m;

    const T block_num_base = (m - 1) / CSRMVN_DIM + 1;

    // if (handle->check)
    // {
    //     if (nnz_per_row < 4)
    //     {
    //         CSRGEMV_VECTOR_ALIGN(2);
    //     }
    //     else if (nnz_per_row < 8)
    //     {
    //         CSRGEMV_VECTOR_ALIGN(4);
    //     }
    //     else if (nnz_per_row < 16)
    //     {
    //         CSRGEMV_VECTOR_ALIGN(8);
    //     }
    //     else if (nnz_per_row < 32)
    //     {
    //         CSRGEMV_VECTOR_ALIGN(16);
    //     }
    //     else
    //     {
    //         CSRGEMV_VECTOR_ALIGN(32);
    //     }

    //     return ALPHA_SPARSE_STATUS_SUCCESS;
    // }

    if (nnz_per_row < 4)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE(2);
    }
    else if (nnz_per_row < 8)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE(4);
    }
    else if (nnz_per_row < 16)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE(8);
    }
    else if (nnz_per_row < 32)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE(16);
    }
    else
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE(32);
    }

    // if (nnz_per_row < 4)
    // {
    //     CSRGEMV_VECTOR_ALIGN(2);
    // }
    // else if (nnz_per_row < 8)
    // {
    //     CSRGEMV_VECTOR_ALIGN(4);
    // }
    // else if (nnz_per_row < 16)
    // {
    //     CSRGEMV_VECTOR_ALIGN(8);
    // }
    // else if (nnz_per_row < 32)
    // {
    //     CSRGEMV_VECTOR_ALIGN(16);
    // }
    // else
    // {
    //     CSRGEMV_VECTOR_ALIGN(32);
    // }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
