#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_coo.h"
#include <cstdint>

#define ITEMS_PER_THREAD 8
#define SPMV_MERGE_BLOCK_SIZE 512
#define BLOCK_SIZE 1024

// cudaEvent_t event_start2, event_stop2;
// float elapsed_time2 = 0.0;

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

template <int block_items, typename T, typename U, typename V, typename W>
__device__ void merge_path_spmv(
    int IPT,
    const T num_rows,
    const T nnz,
    const T num_merge_items,
    const W alpha,
    const U *__restrict__ val,
    const T *__restrict__ col_idxs,
    const T *__restrict__ row_end_ptrs,
    const int srow,
    const U *__restrict__ b,
    const int b_stride,
    const W beta,
    V *__restrict__ c,
    const int c_stride,
    T *__restrict__ row_out,
    U *__restrict__ val_out,
    T *block_start_xs,
    T *block_start_ys)
{
    extern __shared__ char buffer[];

    T block_start_x = block_start_xs[blockIdx.x];
    T block_start_y = block_start_ys[blockIdx.x];
    T end_x = block_start_xs[blockIdx.x + 1];
    T end_y = block_start_ys[blockIdx.x + 1];
    const T block_num_rows = end_x - block_start_x;
    const T block_num_nonzeros = end_y - block_start_y;

    T *shared_row_ptrs = reinterpret_cast<T *>(buffer);
    T *shared_col_idxs = reinterpret_cast<T *>(shared_row_ptrs + block_num_rows);
    U *shared_val = reinterpret_cast<U *>(shared_col_idxs + block_num_nonzeros);

    // 66511.4
    for (int i = threadIdx.x;
         i < block_num_rows && block_start_x + i < num_rows;
         i += SPMV_MERGE_BLOCK_SIZE)
    {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }

    // 995359.6
    for (int i = threadIdx.x;
         i < block_num_nonzeros && block_start_y + i < nnz;
         i += SPMV_MERGE_BLOCK_SIZE)
    {
        shared_val[i] = val[block_start_y + i];
        shared_col_idxs[i] = col_idxs[block_start_y + i];
    }

    cooperative_groups::thread_block g = cooperative_groups::this_thread_block();
    g.sync();

    T start_x;
    T start_y;

    // 266012.4
    merge_path_search(IPT * threadIdx.x, block_num_rows,
                      block_num_nonzeros, shared_row_ptrs, block_start_y,
                      &start_x, &start_y);

    // 1291546.2
    T ind = block_start_y + start_y;
    T row_i = block_start_x + start_x;
    U value = U{};
#pragma unroll
    for (T i = 0; i < IPT && row_i < num_rows; i++)
    {
        if (ind < shared_row_ptrs[start_x] || start_x == block_num_rows)
        {
            value += shared_val[start_y] * __ldg(&b[shared_col_idxs[start_y]]);
            start_y++;
            ind++;
        }
        else
        {
            atomicAdd(&c[row_i], alpha * value);
            start_x++;
            row_i++;
            value = U{};
        }
    }
    g.sync();

    // 140647.7
    const cooperative_groups::thread_block_tile<WARP_SIZE> tile_block =
        cooperative_groups::tiled_partition<WARP_SIZE>(g);
    bool head = segment_scan<WARP_SIZE>(tile_block, row_i, value);
    if (head)
    {
        atomicAdd(&c[row_i], alpha * value);
    }
}

template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void abstract_merge_path_search(
    T *block_start_xs,
    T *block_start_ys,
    T num_rows,
    T nnz,
    const T num_merge_items,
    const T block_items,
    const T block_num,
    const T *row_end_ptrs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid > block_num)
        return;
    const int diagonal =
        min(block_items * gid, num_merge_items);
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, 0,
                      &block_start_xs[gid], &block_start_ys[gid]);
}

template <int block_items, typename T, typename U, typename V, typename W>
__global__ __launch_bounds__(SPMV_MERGE_BLOCK_SIZE) void abstract_merge_path_spmv(
    int items_per_thread,
    const T num_rows,
    const T nnz,
    const T num_merge_items,
    const W __restrict__ alpha,
    const U *__restrict__ val,
    const T *__restrict__ col_idxs,
    const T *__restrict__ row_end_ptrs,
    const int srow,
    const U *__restrict__ b,
    const int b_stride,
    const W __restrict__ beta,
    V *__restrict__ c,
    const int c_stride,
    T *__restrict__ row_out,
    U *__restrict__ val_out,
    T *block_start_xs,
    T *block_start_ys)
{
    merge_path_spmv<block_items>(
        items_per_thread, num_rows, nnz,
        num_merge_items, alpha, val,
        col_idxs, row_end_ptrs, srow, b, b_stride, beta, c, c_stride,
        row_out, val_out, block_start_xs, block_start_ys);
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
    int minimal_num =
        ceildivT(sizeof(T) + sizeof(U), sizeof(T));
    int items_per_thread = ITEMS_PER_THREAD * 4 / sizeof(T);
    items_per_thread = std::max(minimal_num, items_per_thread);

    const T num_merge_items = m + nnz;
    const T block_items = SPMV_MERGE_BLOCK_SIZE * ITEMS_PER_THREAD;
    const T block_num =
        ceildivT(num_merge_items, block_items);

    T *block_start_xs = reinterpret_cast<T *>(externalBuffer);
    T *block_start_ys = reinterpret_cast<T *>(block_start_xs + block_num + 1);
    U *val_out = reinterpret_cast<U *>(block_start_ys + block_num + 1);
    T *row_out = reinterpret_cast<T *>(val_out + block_num);
    int maxbytes = block_items * (sizeof(U) + sizeof(T));
    // printf("maxbytes:%d\n", maxbytes);
    // cudaFuncSetAttribute(abstract_merge_path_spmv<block_items, T, U, V, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    if (&alpha != nullptr && &beta != nullptr)
    {
        abstract_merge_path_search<T><<<ceildivT(block_num, BLOCK_SIZE), BLOCK_SIZE>>>(
            block_start_xs,
            block_start_ys,
            m,
            nnz,
            num_merge_items,
            block_items,
            block_num,
            csr_row_ptr + 1);
        // GPU_TIMER_START(elapsed_time2, event_start2, event_stop2);
        if (block_num > 0)
        {
            array_scale<<<ceildivT(m, BLOCK_SIZE), BLOCK_SIZE>>>(m, y, beta);
            abstract_merge_path_spmv<block_items><<<block_num, SPMV_MERGE_BLOCK_SIZE, maxbytes, 0>>>(
                items_per_thread,
                m,
                nnz,
                num_merge_items,
                alpha,
                csr_val,
                csr_col_ind,
                csr_row_ptr + 1,
                0,
                x,
                1,
                beta,
                y,
                1,
                row_out,
                val_out,
                block_start_xs,
                block_start_ys);
        }
        // GPU_TIMER_END(elapsed_time2, event_start2, event_stop2);
        // printf("compute_time1:%f ms\n", elapsed_time2);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#undef BLOCK_SIZE
#undef ITEMS_PER_THREAD
