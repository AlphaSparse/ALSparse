#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_coo.h"

/**
 * dataset: U.Fl. sparse matrices
 * average speedup(AS) compared with csr-vector.
 * but large ITEMS_PER_THREAD can reduce preprocess time(AVT).
 *
 * ITEMS_PER_THREAD=1, AS: 1.8549
 * ITEMS_PER_THREAD=3, AS: 1.7334, AVT: 0.37ms
 * ITEMS_PER_THREAD=4, AS: 1.6872
 * ITEMS_PER_THREAD=5, AS: 1.5531
 * ITEMS_PER_THREAD=6, AS: 1.5431
 * ITEMS_PER_THREAD=7, AS: 1.4811, AVT: 0.22ms
 * ITEMS_PER_THREAD=8, AS: 1.4088
 * ITEMS_PER_THREAD=9, AS: 1.3422
 */

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 8

struct CoordinateT
{
    long x;
    long y;
};

template <typename T, typename U, typename V, typename W>
static alphasparseStatus_t spmv_csr_merge(alphasparseHandle_t handle,
                                          T m,
                                          T n,
                                          T nnz,
                                          const W alpha,
                                          const U *csr_val,
                                          const T *csr_row_ptr,
                                          const T *csr_col_ind,
                                          const U *x,
                                          const W beta,
                                          V *y);

template <typename T, T BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void get_row_indx_device(T m,
                             const T *csr_row_ptr,
                             T *row_indx)
{
    T tid = blockIdx.x * BLOCK_SIZE + tid;

    if (tid >= m)
        return;

    for (T j = csr_row_ptr[tid]; j < csr_row_ptr[tid + 1]; j++)
    {
        row_indx[j] = tid;
    }
}

// template <typename T, typename V, typename W>
// __launch_bounds__(1024) __global__ static void mulbeta(T m,
//                                                        const W beta,
//                                                        V *__restrict__ y)
// {
//     T tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= m)
//         return;
//     y[tid] *= beta;
//     // alpha_mul(y[tid], y[tid], beta);
// }

// Segmented block reduction kernel
template <typename T, typename U>
static __device__ void segmented_blockreduce(const T *__restrict__ rows, U *__restrict__ vals)
{
    T tid = threadIdx.x;

#pragma unroll
    for (T j = 1; j < BLOCK_THREADS; j <<= 1)
    {
        U val = {};
        if (tid >= j)
        {
            if (rows[tid] == rows[tid - j])
            {
                val = vals[tid - j];
            }
        }
        __syncthreads();

        vals[tid] = vals[tid] + val;
        // alpha_add(vals[tid], vals[tid], val);
        __syncthreads();
    }
}

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <typename T>
__host__ __device__ __forceinline__ static void MergePathSearch(
    T diagonal,
    const T *a,
    // const T *b,
    T a_len,
    T b_len,
    CoordinateT &path_coordinate)
{
    T split_min = max(diagonal - b_len, 0);
    T split_max = min(diagonal, a_len);

    while (split_min < split_max)
    {
        T split_pivot = (split_min + split_max) >> 1;
        // if (a[split_pivot] <= b[diagonal - split_pivot - 1]) {
        if (a[split_pivot] <= diagonal - split_pivot - 1)
        {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        }
        else
        {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    path_coordinate.x = min(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}

/**
 * Spmv search kernel. Identifies merge path starting coordinates for each tile.
 */
template <typename T>
__global__ static void DeviceSpmvSearchKernel(
    int num_merge_tiles,             ///< [in] Number of SpMV merge tiles (spmv grid size)
    CoordinateT *d_tile_coordinates, ///< [out] Pointer to the temporary array of tile starting coordinates
    T num_rows,
    T num_nonzeros,
    const T *d_row_end_offsets) ///< [in] SpMV input parameter bundle
{
    /// Constants
    const T TILE_ITEMS = ITEMS_PER_THREAD;

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < num_merge_tiles + 1)
    {
        T diagonal = (tile_idx * TILE_ITEMS);
        CoordinateT tile_coordinate;

        // Search the merge path
        MergePathSearch(diagonal, d_row_end_offsets, num_rows, num_nonzeros, tile_coordinate);

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;
    }

    if (tile_idx == num_merge_tiles + 1)
    {
        d_tile_coordinates[num_merge_tiles].x = num_rows;
        d_tile_coordinates[num_merge_tiles].y = num_nonzeros;
    }
}

/**
 * Spmv kernel.
 */
template <typename T, typename U, typename V, typename W>
__global__ static void DeviceSpmvKernel(T m,
                                        W alpha,
                                        const T *csr_row_ptr,
                                        const T *csr_col_ind,
                                        const U *csr_val,
                                        const U *x,
                                        W beta,
                                        V *y,
                                        T num_merge_tiles,
                                        const CoordinateT *path_coordinate,
                                        U *reduc_val,
                                        T *reduc_row)
{
    const T bid = blockIdx.x;
    const T gid = (blockIdx.x * BLOCK_THREADS) + threadIdx.x;
    const T tid = threadIdx.x;

    __shared__ T s_row[BLOCK_THREADS];
    __shared__ U s_val[BLOCK_THREADS];
    __shared__ U s_val2[BLOCK_THREADS * ITEMS_PER_THREAD];
    __shared__ T s_col_ind2[BLOCK_THREADS * ITEMS_PER_THREAD];

    if (gid >= num_merge_tiles)
        return;

    T row_idx_start = path_coordinate[gid].x;
    T row_idx_end = path_coordinate[gid + 1].x;

    T ai = path_coordinate[gid].y;
    const T ai_end = path_coordinate[gid + 1].y;

    const T gid2 = blockIdx.x * BLOCK_THREADS;
    const T gid3 = (blockIdx.x + 1) * BLOCK_THREADS;
    const T ai2 = path_coordinate[gid2].y;
    T ai_end2;
    if (gid3 >= num_merge_tiles)
    {
        ai_end2 = path_coordinate[num_merge_tiles].y;
    }
    else
    {
        ai_end2 = path_coordinate[gid3].y;
    }

    s_val[tid] = U{};
    s_row[tid] = -1;
    for (int i = ai2; i + tid < ai_end2; i += BLOCK_THREADS)
    {
        s_val2[tid + i - ai2] = csr_val[i + tid];
        s_col_ind2[tid + i - ai2] = csr_col_ind[i + tid];
    }
    __syncthreads();

    V sum;
    T row = row_idx_start;
    for (; row < row_idx_end; row++)
    {
        sum = V{};
        T row_end4 = csr_row_ptr[row + 1] - 3;
        for (; ai < row_end4; ai += 4)
        {
            int temp = ai - ai2;
            sum += s_val2[temp] * x[s_col_ind2[temp]];
            sum += s_val2[temp + 1] * x[s_col_ind2[temp + 1]];
            sum += s_val2[temp + 2] * x[s_col_ind2[temp + 2]];
            sum += s_val2[temp + 3] * x[s_col_ind2[temp + 3]];
        }
        for (; ai < csr_row_ptr[row + 1]; ai++)
        {
            int temp = ai - ai2;
            sum += s_val2[temp] * x[s_col_ind2[temp]];
        }
        y[row] += sum * alpha;
    }

    sum = V{};
    T ai_end4 = ai_end - 3;
    for (; ai < ai_end4; ai += 4)
    {
        int temp = ai - ai2;
        sum += s_val2[temp] * x[s_col_ind2[temp]];
        sum += s_val2[temp + 1] * x[s_col_ind2[temp + 1]];
        sum += s_val2[temp + 2] * x[s_col_ind2[temp + 2]];
        sum += s_val2[temp + 3] * x[s_col_ind2[temp + 3]];
    }
    for (; ai < ai_end; ai++)
    {
        int temp = ai - ai2;
        sum += s_val2[temp] * x[s_col_ind2[temp]];
    }
    sum *= alpha;
    // alpha_mule(sum, alpha);
    s_val[tid] += sum;
    // alpha_adde(s_val[tid], sum);
    s_row[tid] = row;
    __syncthreads();

    //! inner block reduction 1
    // Segmented block reduction
    for (T j = 1; j < BLOCK_THREADS; j <<= 1)
    {
        if (tid >= j)
        {
            if (row == s_row[tid - j])
            {
                sum = sum + s_val[tid - j];
                // alpha_add(sum, sum, s_val[tid - j]);
            }
        }
        __syncthreads();

        s_val[tid] = sum;

        __syncthreads();
    }
    // All thread but the last one write their result in y.
    if (tid < BLOCK_THREADS - 1)
    {
        if (row != s_row[tid + 1] && row >= 0)
        {
            y[row] = y[row] + sum;
            // alpha_adde(y[row], sum);
        }
    }

    // for inter block reduction
    if (tid == BLOCK_THREADS - 1)
    {
        //! alpha_mule(y[row], beta);
        reduc_val[bid] = sum;
        reduc_row[bid] = row;
    }
}

// Do the final block reduction of the block reduction buffers back into global memory
template <typename T, typename U, typename V>
__launch_bounds__(BLOCK_THREADS) __global__ static void csrmv_merge_general_block_reduce(
    T nnz,
    T m,
    const T *__restrict__ row_block_red,
    const U *__restrict__ val_block_red,
    V *__restrict__ y)
{
    T tid = threadIdx.x;

    // Quick return when thread is out of bounds
    if (tid >= nnz)
    {
        return;
    }

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ T shared_row[BLOCK_THREADS];
    __shared__ U shared_val[BLOCK_THREADS];

    // Loop over blocks that are subject for segmented reduction
    for (T i = tid; i < nnz; i += BLOCK_THREADS)
    {
        // Copy data to reduction buffers
        shared_row[tid] = row_block_red[i];
        shared_val[tid] = val_block_red[i];

        __syncthreads();

        // Do segmented block reduction
        segmented_blockreduce(shared_row, shared_val);

        // Add reduced sum to y if valid
        T row = shared_row[tid];
        T rowp1 = (tid < BLOCK_THREADS - 1) ? shared_row[tid + 1] : -1;

        if (row != rowp1 && row >= 0 && row < m)
        {
            y[row] = y[row] + shared_val[tid];
            // alpha_add(y[row], y[row], shared_val[tid]);
        }

        __syncthreads();
    }
}

/**
 * merge-based csrmv: merge based load balance.
 * Merrill D, Garland M.
 * Merge-based sparse matrix-vector multiplication (spmv) using the csr storage format
 * ACM SIGPLAN Notices, 2016, 51(8): 1-2.
 *
 * worktiems: (m + 1) + nnz, row_ptr + values.
 * csr2coo can achieve similar results.
 *
 * if 1 item per thread, same as coo, merge path can convert rowptr to rowindx fast
 *
 * balance memory access: coo & csr
 * sizeof(row_ptr) + sizeof(val) + len(col_ind) vs len(row_ind) + len(val) + len(col_ind)
 *
 * balance calculate: coo & csr
 * calculate item for each wavefront.
 * segment reduce & block reduce for coo vs wavefront reduce for csr
 *
 */
template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_merge(alphasparseHandle_t handle,
                                   T m,
                                   T n,
                                   T nnz,
                                   const W alpha,
                                   const U *csr_val,
                                   const T *csr_row_ptr,
                                   const T *csr_col_ind,
                                   const U *x,
                                   const W beta,
                                   V *y,
                                   void *externalBuffer)
{
    // // dispatch coo case
    // if (ITEMS_PER_THREAD == 1)
    // {
    //     return spmv_csr_merge(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    // }

    if (n == 1)
    {
        return spmv_csr_scalar(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }

    const T work_item = m + nnz;
    const T num_merge_tiles = CEIL(work_item, ITEMS_PER_THREAD);
    const T num_blocks = CEIL(num_merge_tiles, BLOCK_THREADS);

    if (num_blocks <= 1)
    {
        return spmv_csr_vector(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }

    CoordinateT *d_tile_coordinates = nullptr;
    T *reduc_row = nullptr;
    U *reduc_val = nullptr;

    printf("m:%d n:%d num_merge_tiles:%d, ITEMS_PER_THREAD:%d\n", m, nnz, num_merge_tiles, ITEMS_PER_THREAD);

    hipMalloc(&d_tile_coordinates, sizeof(CoordinateT) * (num_merge_tiles + 1));
    double time = get_time_us();
    hipLaunchKernelGGL(DeviceSpmvSearchKernel, dim3(CEIL(num_merge_tiles, 256)), dim3(256), 0, handle->stream, num_merge_tiles, d_tile_coordinates, m, nnz, csr_row_ptr + 1);
    time = (get_time_us() - time) / (1e3);
    printf("preprocess:%f\n", time);
    hipMalloc(&reduc_row, sizeof(T) * num_blocks);
    hipMalloc(&reduc_val, sizeof(U) * num_blocks);

    hipLaunchKernelGGL(mulbeta, dim3(CEIL(m, 1024)), dim3(1024), 0, handle->stream, m, beta, y);
    time = get_time_us();
    hipLaunchKernelGGL(DeviceSpmvKernel, dim3(num_blocks), dim3(BLOCK_THREADS), 0, handle->stream, m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, num_merge_tiles, d_tile_coordinates, reduc_val, reduc_row);
    time = (get_time_us() - time) / (1e3);
    printf("spmv:%f\n", time);
    time = get_time_us();
    hipLaunchKernelGGL(csrmv_merge_general_block_reduce, dim3(1), dim3(BLOCK_THREADS), 0, handle->stream, num_blocks, m, reduc_row, reduc_val, y);
    time = (get_time_us() - time) / (1e3);
    printf("reduce:%f\n", time);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

// /**
//  *  coo case for ITEMS_PER_THREAD == 1
//  *
//  */
// template <typename T, typename U, typename V, typename W>
// static alphasparseStatus_t spmv_csr_merge(alphasparseHandle_t handle,
//                                           T m,
//                                           T n,
//                                           T nnz,
//                                           const W alpha,
//                                           const U *csr_val,
//                                           const T *csr_row_ptr,
//                                           const T *csr_col_ind,
//                                           const U *x,
//                                           const W beta,
//                                           V *y)
// {
//     T *d_row_indx;
//     hipMalloc(&d_row_indx, sizeof(T) * nnz);

//     const T BLOCK_SIZE = 512;
//     hipLaunchKernelGGL(HIP_KERNEL_NAME(get_row_indx_device<BLOCK_SIZE>), dim3(CEIL(m, BLOCK_SIZE)), dim3(BLOCK_SIZE), 0, handle->stream, m, csr_row_ptr, d_row_indx);
//     void* coordinate = (void *)d_row_indx;
//     spmv_coo(handle,
//                  m,
//                  n,
//                  nnz,
//                  alpha,
//                  csr_val,
//                  coordinate,
//                  csr_col_ind,
//                  x,
//                  beta,
//                  y);

//     return ALPHA_SPARSE_STATUS_SUCCESS;
// }

#undef BLOCK_THREADS
#undef ITEMS_PER_THREAD
