#pragma once

#include "alphasparse.h"

#define ITEMS_PER_THREAD 8
#define SPMV_MERGE_BLOCK_SIZE 512
#define BLOCK_SIZE 512

// cudaEvent_t event_start2, event_stop2;
// float elapsed_time2 = 0.0;

template <typename T>
__forceinline__ __device__ void merge_path_search(
    const T diagonal, const T x_len, const T y_len,
    const T *__restrict__ a, const T offset_y,
    T *__restrict__ x, T *__restrict__ y)
{
    T x_min = max(diagonal - y_len, 0);
    T x_max = min(diagonal, x_len);
    T pivot;
    while (x_min < x_max)
    {
        pivot = (x_max + x_min) / 2;
        if (a[pivot / 2] < offset_y + diagonal - pivot)
        {
            x_min = pivot + 1;
        }
        else
        {
            x_max = pivot;
        }
    }
    *x = x_min;
    *y = diagonal - x_min;
}

template <int IPT, typename T, typename U, typename V, typename W>
__global__ __launch_bounds__(SPMV_MERGE_BLOCK_SIZE) void merge_path_spmv(
    const T num_rows,
    const T nnz,
    const T num_merge_items,
    const W alpha,
    const U *__restrict__ val,
    const T *__restrict__ col_idxs,
    const T *__restrict__ row_end_ptrs,
    const U *__restrict__ x,
    const W beta,
    V *__restrict__ y,
    T *block_start_xs,
    T *block_start_ys)
{
    extern __shared__ char buffer[];

    T block_start_x2 = block_start_xs[blockIdx.x];
    T block_start_x = block_start_x2 / 2;
    const T block_start_y = block_start_ys[blockIdx.x];
    const T block_num_rows2 = block_start_xs[blockIdx.x + 1] - block_start_x2;
    const T block_num_rows = block_num_rows2 / 2;
    const T block_num_nnz = block_start_ys[blockIdx.x + 1] - block_start_y;
    T *shared_row_ptrs = reinterpret_cast<T *>(buffer);
    T *shared_col_idxs = reinterpret_cast<T *>(shared_row_ptrs + block_num_rows + 1);
    U *shared_val = reinterpret_cast<U *>(shared_col_idxs + block_num_nnz);
    // if (block_start_x > 101316)
    //     printf("block_num_rows:%d, block_num_nnz:%d, block_start_x:%d, block_start_y:%d\n", block_num_rows2, block_num_nnz, block_start_x, block_start_y);
    // if (block_num_rows2 + block_num_nnz != 4096)
    //     printf("block_num_rows:%d, block_num_nnz:%d, block_start_x:%d, block_start_y:%d\n", block_num_rows2, block_num_nnz, block_start_x, block_start_y);
    // 66511.4
    for (int i = threadIdx.x; i <= block_num_rows; i += SPMV_MERGE_BLOCK_SIZE)
    {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }

    // 995359.6
    for (int i = threadIdx.x; i < block_num_nnz; i += SPMV_MERGE_BLOCK_SIZE)
    {
        shared_val[i] = val[block_start_y + i];
        shared_col_idxs[i] = col_idxs[block_start_y + i];
    }

    cooperative_groups::thread_block g = cooperative_groups::this_thread_block();
    g.sync();

    if ((blockIdx.x * blockDim.x + threadIdx.x) * IPT >= num_rows + nnz)
    {
        return;
    }

    T start_x;
    T start_y;

    // 266012.4
    merge_path_search<T>(IPT * threadIdx.x, block_num_rows2,
                         block_num_nnz, shared_row_ptrs, block_start_y,
                         &start_x2, &start_y);
    // start_x2 = (start_x2 + 1) / 2 * 2;
    // 1291546.2
    T row_i2 = block_start_x2 + start_x2;
    T ind = block_start_y + start_y;
    U value = U{};
    T temp_row_ptr = shared_row_ptrs[start_x];
#pragma unroll
    for (T i = 0; i < IPT && row_i < num_rows && start_y < block_num_nnz; i++)
    {
        // if (start_x >= block_num_rows)
        // {
        //     printf("start_x:%d, block_num_rows:%d, start_x2:%d, block_num_rows2:%d\n", start_x, block_num_rows, start_x2, block_num_rows2);
        // }
        // if (start_y >= block_num_nnz)
        // {
        //     printf("start_y:%d, block_num_nnz:%d\n", start_y, block_num_nnz);
        // }
        if (ind < temp_row_ptr || start_x == block_num_rows)
        {

        // if (start_x2 % 2 == 1)
        // {
        //     start_x2++;
        //     row_i2++;
        //     continue;
        // }

        // if (blockIdx.x == 1)
        //     printf("start_x:%d, start_y:%d, row_i:%d, i:%d, block_num_rows2:%d\n", start_x, start_y, row_i, i, block_num_rows2);
        // if (row_i == 857)
        // {
        //     printf("i:%d, tid:%d, bid:%d, block_start_x:%d, block_start_y:%d, y[row_i]:%f, value:%f, block_num_rows2:%d, row_i2:%d, ind:%d, start_x2:%d, start_y:%d, block_num_nnz:%d, start_x:%d, row_i:%d, block_num_rows:%d, num_rows:%d, shared_val[start_y]:%f, shared_col_idxs[start_y]:%d, shared_row_ptrs[start_x]:%d\n",
        //            i, threadIdx.x, blockIdx.x, block_start_x, block_start_y, y[row_i], value, block_num_rows2, row_i2, ind, start_x2, start_y, block_num_nnz, start_x, row_i, block_num_rows, num_rows, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
        // }
        // if (row_i == 857)
        // {
        //     printf("ind:%d, shared_row_ptrs[start_x]:%d\n",
        //            ind, shared_row_ptrs[start_x]);
        // }
        if (ind < shared_row_ptrs[start_x])
        {
            value += shared_val[start_y] * __ldg(&x[shared_col_idxs[start_y]]);
            start_y++;
            ind++;
        }
        else
        {
            // if (row_i == 1)
            // {
            //     // printf("start_y:%d, \n",
            //     //        start_y, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
            //     printf("tid:%d, y[row_i]:%f, value:%f, block_num_rows2:%d, row_i2:%d, ind:%d, start_x2:%d, start_y:%d, block_num_nnz:%d, start_x:%d, row_i:%d, block_num_rows:%d, num_rows:%d, shared_val[start_y]:%f, shared_col_idxs[start_y]:%d, shared_row_ptrs[start_x]:%d\n",
            //            threadIdx.x, y[row_i], value, block_num_rows2, row_i2, ind, start_x2, start_y, block_num_nnz, start_x, row_i, block_num_rows, num_rows, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
            // }./
            atomicAdd(&y[row_i], alpha * value);
            start_x++;
            row_i++;
            temp_row_ptr = shared_row_ptrs[start_x];
        }
        // if (row_i == 2)
        // {
        //     printf("i:%d, tid:%d, bid:%d, y[row_i-1]:%f, value:%f, block_num_rows2:%d, row_i2:%d, ind:%d, start_x2:%d, start_y:%d, block_num_nnz:%d, start_x:%d, row_i:%d, block_num_rows:%d, num_rows:%d, shared_val[start_y]:%f, shared_col_idxs[start_y]:%d, shared_row_ptrs[start_x]:%d\n",
        //            i, threadIdx.x, blockIdx.x, y[row_i - 1], value, block_num_rows2, row_i2, ind, start_x2, start_y, block_num_nnz, start_x, row_i, block_num_rows, num_rows, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
        // }
        // if (row_i == 1)
        // {
        //     printf("tid:%d, value:%f, block_num_rows2:%d, row_i2:%d, ind:%d, start_x2:%d, start_y:%d, block_num_nnz:%d, start_x:%d, row_i:%d, block_num_rows:%d, num_rows:%d, shared_val[start_y]:%f, shared_col_idxs[start_y]:%d, shared_row_ptrs[start_x]:%d\n",
        //            threadIdx.x, value, block_num_rows2, row_i2, ind, start_x2, start_y, block_num_nnz, start_x, row_i, block_num_rows, num_rows, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
        // }
    }
    // 140647.7
    const cooperative_groups::thread_block_tile<WARP_SIZE> tile_block =
        cooperative_groups::tiled_partition<WARP_SIZE>(g);
    bool head = segment_scan<WARP_SIZE>(tile_block, row_i, value);
    // if (row_i == 5)
    // {
    //     // printf("start_y:%d, \n",
    //     //        start_y, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
    //     printf("tid:%d, y[row_i]:%f, value:%f, block_num_rows2:%d, row_i2:%d, ind:%d, start_x2:%d, start_y:%d, block_num_nnz:%d, start_x:%d, row_i:%d, block_num_rows:%d, num_rows:%d, shared_val[start_y]:%f, shared_col_idxs[start_y]:%d, shared_row_ptrs[start_x]:%d\n",
    //            threadIdx.x, y[row_i], value, block_num_rows2, row_i2, ind, start_x2, start_y, block_num_nnz, start_x, row_i, block_num_rows, num_rows, shared_val[start_y], shared_col_idxs[start_y], shared_row_ptrs[start_x]);
    // }
    if (head)
    {
        atomicAdd(&y[row_i], alpha * value);
    }
}

template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void abstract_merge_path_search(
    T *block_start_xs,
    T *block_start_ys,
    T num_rows,
    T nnz,
    const T num_merge_items,
    const T items_per_block,
    const T block_num,
    const T *row_end_ptrs)
{
    const T gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid > block_num)
    {
        if (gid < num_rows + block_num + 1)
        {
            y[gid - block_num - 1] *= beta;
        }
        return;
    }
    const T diagonal = min(items_per_block * gid, num_merge_items);
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, 0,
                      &block_start_xs[gid], &block_start_ys[gid]);
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
    const T num_merge_items = m + nnz;
    const T items_per_block = SPMV_MERGE_BLOCK_SIZE * ITEMS_PER_THREAD;
    const T block_num = ceildivT(num_merge_items, items_per_block);

    T *block_start_xs = reinterpret_cast<T *>(externalBuffer);
    T *block_start_ys = reinterpret_cast<T *>(block_start_xs + block_num + 1);
    const int maxbytes = items_per_block * (sizeof(U) + sizeof(T)) + sizeof(T);
    printf("block_num:%d\n", block_num);
    // printf("maxbytes:%d\n", maxbytes);
    // cudaFuncSetAttribute(abstract_merge_path_spmv<items_per_block, T, U, V, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    if (&alpha != nullptr && &beta != nullptr)
    {
        abstract_merge_path_search<T><<<ceildivT(block_num, BLOCK_SIZE), BLOCK_SIZE>>>(
            block_start_xs,
            block_start_ys,
            m * 2,
            nnz,
            num_merge_items,
            items_per_block,
            block_num,
            csr_row_ptr + 1);
        // T *block_start_xs_h = (T *)malloc(sizeof(T) * (block_num + 1));
        // T *block_start_ys_h = (T *)malloc(sizeof(T) * (block_num + 1));
        // cudaMemcpy(block_start_xs_h, block_start_xs, sizeof(T) * (block_num + 1), cudaMemcpyDeviceToHost);
        // cudaMemcpy(block_start_ys_h, block_start_ys, sizeof(T) * (block_num + 1), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < block_num + 1; i++)
        // {
        //     printf("x: %d, y: %d\n", block_start_xs_h[i] / 2, block_start_ys_h[i]);
        // }
        // GPU_TIMER_START(elapsed_time2, event_start2, event_stop2);
        if (block_num > 0)
        {
            merge_path_spmv<ITEMS_PER_THREAD><<<block_num, SPMV_MERGE_BLOCK_SIZE, maxbytes, 0>>>(
                m,
                nnz,
                num_merge_items,
                alpha,
                csr_val,
                csr_col_ind,
                csr_row_ptr + 1,
                x,
                beta,
                y,
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
