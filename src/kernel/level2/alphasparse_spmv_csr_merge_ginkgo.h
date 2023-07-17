#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_coo.h"
#include <cooperative_groups.h>
#include <cstdint>

#define ITEMS_PER_THREAD 8
#define SPMV_BLOCK_SIZE 128

cudaEvent_t event_start2, event_stop2;
float elapsed_time2 = 0.0;

inline constexpr int64_t ceildiv(int64_t num, int64_t den)
{
    return (num + den - 1) / den;
}

template <typename T, typename U, typename V, typename W>
static alphasparseStatus_t spmv_csr_merge_ginkgo(alphasparseHandle_t handle,
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

template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool block_segment_scan_reverse(
    const IndexType *__restrict__ ind, ValueType *__restrict__ val)
{
    bool last = true;
    const auto reg_ind = ind[threadIdx.x];
#pragma unroll
    for (int i = 1; i < SPMV_BLOCK_SIZE; i <<= 1)
    {
        if (i == 1 && threadIdx.x < SPMV_BLOCK_SIZE - 1 &&
            reg_ind == ind[threadIdx.x + 1])
        {
            last = false;
        }
        auto temp = ValueType{};
        if (threadIdx.x >= i && reg_ind == ind[threadIdx.x - i])
        {
            temp = val[threadIdx.x - i];
        }
        cooperative_groups::this_thread_block().sync();
        val[threadIdx.x] += temp;
        cooperative_groups::this_thread_block().sync();
    }

    return last;
}

template <typename T, typename U, typename V, typename W>
__global__ void merge_path_reduce(const int nwarps,
                                  const U *__restrict__ last_val,
                                  const T *__restrict__ last_row,
                                  V *__restrict__ c,
                                  const int c_stride, const W alpha)
{
    const int cache_lines = ceildivT<int>(nwarps, SPMV_BLOCK_SIZE);
    const int tid = threadIdx.x;
    const int start = min(tid * cache_lines, nwarps);
    const int end = min((tid + 1) * cache_lines, nwarps);
    U value = U{};
    int row = last_row[nwarps - 1];
    if (start < nwarps)
    {
        value = __ldg(&last_val[start]);
        row = __ldg(&last_row[start]);
        for (int i = start + 1; i < end; i++)
        {
            if (__ldg(&last_row[i]) != row)
            {
                c[row] += alpha * (value);
                row = __ldg(&last_row[i]);
                value = __ldg(&last_val[i]);
            }
            else
            {
                value += __ldg(&last_val[i]);
            }
        }
    }
    __shared__ int tmp_ind[SPMV_BLOCK_SIZE];
    __shared__ U tmp_val[SPMV_BLOCK_SIZE];
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row;
    cooperative_groups::this_thread_block().sync();
    bool last = block_segment_scan_reverse(static_cast<T *>(tmp_ind),
                                           static_cast<U *>(tmp_val));
    cooperative_groups::this_thread_block().sync();
    if (last)
    {
        c[row] += alpha * (tmp_val[threadIdx.x]);
    }
}

__forceinline__ __device__ void merge_path_search(
    const int diagonal, const int a_len, const int b_len,
    const int *__restrict__ a, const int offset_b,
    int *__restrict__ x, int *__restrict__ y)
{
    auto x_min = max(diagonal - b_len, 0);
    auto x_max = min(diagonal, a_len);
    while (x_min < x_max)
    {
        auto pivot = x_min + (x_max - x_min) / 2;
        if (a[pivot] <= offset_b + diagonal - pivot - 1)
        {
            x_min = pivot + 1;
        }
        else
        {
            x_max = pivot;
        }
    }

    *x = min(x_min, a_len);
    *y = diagonal - x_min;
}

template <typename T, typename U, typename V, typename W>
__device__ void merge_path_spmv(
    int IPT,
    const T num_rows, const T nnz, const T num_merge_items,
    const T block_items, const W alpha, const U *__restrict__ val,
    const T *__restrict__ col_idxs,
    const T *__restrict__ row_ptrs, const int srow,
    const U *__restrict__ b, const int b_stride, const W beta,
    V *__restrict__ c, const int c_stride,
    T *__restrict__ row_out, U *__restrict__ val_out)
{
    const auto *row_end_ptrs = row_ptrs + 1;
    extern __shared__ char buffer[];
    T *shared_row_ptrs = reinterpret_cast<T *>(buffer);
    U *shared_val = reinterpret_cast<U *>(&buffer[block_items * sizeof(U)]);
    T *shared_col_idxs = reinterpret_cast<T *>(&buffer[block_items * (sizeof(U) + sizeof(T))]);

    const int diagonal =
        min(block_items * blockIdx.x, num_merge_items);
    const int diagonal_end = min(diagonal + block_items, num_merge_items);
    T block_start_x;
    T block_start_y;
    T end_x;
    T end_y;
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, 0,
                      &block_start_x, &block_start_y);
    merge_path_search(diagonal_end, num_rows, nnz, row_end_ptrs,
                      0, &end_x, &end_y);
    const T block_num_rows = end_x - block_start_x;
    const T block_num_nonzeros = end_y - block_start_y;
    for (int i = threadIdx.x;
         i < block_num_rows && block_start_x + i < num_rows;
         i += SPMV_BLOCK_SIZE)
    {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }
    for (int i = threadIdx.x;
         i < block_num_nonzeros && block_start_y + i < nnz;
         i += SPMV_BLOCK_SIZE)
    {
        shared_val[i] = val[block_start_y + i];
        shared_col_idxs[i] = col_idxs[block_start_y + i];
    }
    cooperative_groups::this_thread_block().sync();

    T start_x;
    T start_y;
    merge_path_search(int(IPT * threadIdx.x), block_num_rows,
                      block_num_nonzeros, shared_row_ptrs, block_start_y,
                      &start_x, &start_y);

    T ind = block_start_y + start_y;
    T row_i = block_start_x + start_x;
    U value = U{};
#pragma unroll
    for (T i = 0; i < IPT; i++)
    {
        if (row_i < num_rows)
        {
            if (start_x == block_num_rows || ind < shared_row_ptrs[start_x])
            {
                value += shared_val[ind - block_start_y] * __ldg(&b[shared_col_idxs[ind - block_start_y]]);
                ind++;
            }
            else
            {
                c[row_i] = alpha * value + beta * c[row_i];
                start_x++;
                row_i++;
                value = U{};
            }
        }
    }
    cooperative_groups::this_thread_block().sync();
    int *tmp_ind = shared_row_ptrs;
    U *tmp_val =
        reinterpret_cast<U *>(shared_row_ptrs + SPMV_BLOCK_SIZE);
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row_i;
    cooperative_groups::this_thread_block().sync();
    bool last = block_segment_scan_reverse(tmp_ind, tmp_val);
    if (threadIdx.x == SPMV_BLOCK_SIZE - 1)
    {
        row_out[blockIdx.x] = min(end_x, num_rows - 1);
        val_out[blockIdx.x] = tmp_val[threadIdx.x];
    }
    else if (last)
    {
        c[row_i] += alpha * tmp_val[threadIdx.x];
    }
}

template <typename T, typename U, typename V, typename W>
__global__ __launch_bounds__(SPMV_BLOCK_SIZE) void abstract_merge_path_spmv(
    int items_per_thread,
    const T num_rows, const T nnz, const T num_merge_items,
    const int block_items, const W __restrict__ alpha,
    const U *__restrict__ val, const T *__restrict__ col_idxs,
    const T *__restrict__ row_ptrs, const int srow,
    const U *__restrict__ b, const int b_stride,
    const W __restrict__ beta, V *__restrict__ c,
    const int c_stride, T *__restrict__ row_out,
    U *__restrict__ val_out)
{
    merge_path_spmv(
        items_per_thread, num_rows, nnz,
        num_merge_items, block_items, alpha, val,
        col_idxs, row_ptrs, srow, b, b_stride, beta, c, c_stride,
        row_out, val_out);
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
alphasparseStatus_t spmv_csr_merge_ginkgo(alphasparseHandle_t handle,
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
    if (n == 1)
    {
        return spmv_csr_scalar(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }
    T *row_out = nullptr;
    U *val_out = nullptr;
    constexpr int minimal_num =
        ceildiv(sizeof(T) + sizeof(U), sizeof(T));
    int items_per_thread = ITEMS_PER_THREAD * 4 / sizeof(T);
    items_per_thread = std::max(minimal_num, items_per_thread);

    const T num_merge_items = m + nnz;
    const T block_items = SPMV_BLOCK_SIZE * items_per_thread;
    const T grid_num =
        ceildiv(num_merge_items, block_items);
    const auto grid = grid_num;

    val_out = (U *)externalBuffer;
    row_out = reinterpret_cast<T *>(val_out + grid_num);
    int maxbytes = block_items * (sizeof(U) + sizeof(T) * 2);
    // printf("maxbytes:%d\n", maxbytes);
    cudaFuncSetAttribute(abstract_merge_path_spmv<T, U, V, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    if (&alpha != nullptr && &beta != nullptr)
    {
        // GPU_TIMER_START(elapsed_time2, event_start2, event_stop2);
        if (grid_num > 0)
        {
            abstract_merge_path_spmv<<<grid, SPMV_BLOCK_SIZE, maxbytes, 0>>>(
                items_per_thread,
                m,
                nnz,
                num_merge_items,
                block_items,
                alpha,
                csr_val,
                csr_col_ind,
                csr_row_ptr,
                0,
                x,
                1,
                beta,
                y,
                1,
                row_out,
                val_out);
        }
        // GPU_TIMER_END(elapsed_time2, event_start2, event_stop2);
        // printf("compute_time1:%f ms\n", elapsed_time2);
        // GPU_TIMER_START(elapsed_time2, event_start2, event_stop2);
        merge_path_reduce<<<1, SPMV_BLOCK_SIZE, 0, 0>>>(
            grid_num, val_out,
            row_out,
            y,
            1, alpha);
        // GPU_TIMER_END(elapsed_time2, event_start2, event_stop2);
        // printf("compute_time2:%f ms\n", elapsed_time2);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#undef ITEMS_PER_THREAD
