#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_csr_vector.h"
#include <iostream>
template <typename T>
__device__ static T lower_bound_int_acc(const T *t, T l, T r, T target)
{
    while (r > l)
    {
        T m = (l + r) / 2 + 1;
        if (t[m] <= target)
        {
            l = m;
        }
        else
        {
            r = m - 1;
        }
    }
    return l;
}

template <typename T, typename W, typename V>
__global__ static void balanced_partition_row_by_nnz_acc_and_scale_y(const T *acc_sum_arr, T rows, T nwarps, T *partition, T nnz, T nnz_per_block, W beta, V *y)
{
    const T gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid > nwarps)
    {
        if (gid <= nwarps + rows)
        {
            y[gid - nwarps - 1] *= beta;
        }
        return;
    }
    partition[gid] = lower_bound_int_acc(acc_sum_arr, 0, rows, (nnz_per_block * gid));
}

template <typename T>
__global__ static void balanced_partition_row_by_nnz_acc(const T *acc_sum_arr, T rows, T *partition, T nnz_per_block)
{
    const T gid = blockIdx.x * blockDim.x + threadIdx.x;
    partition[gid] = lower_bound_int_acc(acc_sum_arr, 0, rows, (nnz_per_block * gid));
}

template <typename T, typename U, typename V, typename W>
__device__ void flat_reduce_direct2(
    const W alpha,
    const T *__restrict__ csr_row_ptr,
    U *shared_data,
    V *__restrict__ y,
    const T block_start_row,
    T block_end_row,
    const T block_start_ind,
    const T block_num_nnz)
{
    for (int row_i = block_start_row + threadIdx.x; row_i < block_end_row; row_i += blockDim.x)
    {
        T ind_start = max(csr_row_ptr[row_i] - block_start_ind, 0);
        T ind_end = min(csr_row_ptr[row_i + 1] - block_start_ind, block_num_nnz);
        U sum = U{};
        for (int i = ind_start; i < ind_end; i++)
        {
            sum += shared_data[i];
        }
        atomicAdd(&y[row_i], alpha * sum);
    }
}

template <typename T, typename U, typename V, typename W, int VECTOR_SIZE>
__device__ void flat_reduce_in_vector2(
    const W alpha,
    const T *__restrict__ csr_row_ptr,
    U *shared_data,
    V *__restrict__ y,
    const T block_start_row,
    T block_end_row,
    const T block_start_ind,
    const T block_num_nnz)
{
    const int vec_num = blockDim.x / VECTOR_SIZE;
    const int vec_id = threadIdx.x / VECTOR_SIZE;
    const int tid_in_vec = threadIdx.x % VECTOR_SIZE;

    for (int row_i = block_start_row + vec_id; row_i < block_end_row; row_i += vec_num)
    {
        T ind_start = max(csr_row_ptr[row_i] - block_start_ind, 0);
        T ind_end = min(csr_row_ptr[row_i + 1] - block_start_ind, block_num_nnz);
        U sum = U{};
        for (int i = ind_start + tid_in_vec; i < ind_end; i += VECTOR_SIZE)
        {
            sum += shared_data[i];
        }
        for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_down(sum, i, VECTOR_SIZE);
        }
        // store value
        if (tid_in_vec == 0)
        {
            atomicAdd(&y[row_i], alpha * sum);
        }
    }
}

template <typename T, typename U, typename V, typename W, int BLOCK_SIZE, int VECTOR_SIZE>
__device__ void flat_reduce_in_vector_with_mem_coalescing2(
    const W alpha,
    const T *__restrict__ csr_row_ptr,
    U *shared_data,
    V *__restrict__ y,
    const T block_start_row,
    T block_end_row,
    const T block_start_ind,
    const T block_num_nnz)
{
    const int vec_num = blockDim.x / VECTOR_SIZE;
    const int vec_id = threadIdx.x / VECTOR_SIZE;
    const int tid_in_vec = threadIdx.x % VECTOR_SIZE;
    __shared__ V lds_y[BLOCK_SIZE / VECTOR_SIZE];

    for (int row_i = block_start_row + vec_id; row_i < block_end_row; row_i += vec_num)
    {
        T ind_start = max(csr_row_ptr[row_i] - block_start_ind, 0);
        T ind_end = min(csr_row_ptr[row_i + 1] - block_start_ind, block_num_nnz);
        U sum = U{};
        for (int i = ind_start + tid_in_vec; i < ind_end; i += VECTOR_SIZE)
        {
            sum += shared_data[i];
        }
        for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_down(sum, i, VECTOR_SIZE);
        }
        // store value
        if (tid_in_vec == 0)
        {
            lds_y[vec_id] = sum;
        }
    }
    __syncthreads();
    // T thread_reduce_row_id = block_start_row + threadIdx.x;
    // // store sum value to y with memory coalescing
    // if (threadIdx.x < vec_num && thread_reduce_row_id < block_end_row)
    // {
    //     const U local_sum = lds_y[threadIdx.x];
    //     atomicAdd(&y[thread_reduce_row_id], alpha * local_sum);
    // }
}

template <typename T, typename U, typename V, typename W, int BLOCK_SIZE, T nnz_per_block, int REDUCE_OPTION, int R>
__global__ void spmv_acc(
    const T m,
    const T nnz,
    const W alpha,
    const T *__restrict__ partition,
    const T *__restrict__ csr_row_ptr,
    const T *__restrict__ csr_col_ind,
    const U *__restrict__ csr_val,
    const U *__restrict__ x,
    const W beta,
    V *__restrict__ y)
{
    __shared__ U shared_data[nnz_per_block];
    const T block_start_row = partition[blockIdx.x];
    T block_end_row = partition[blockIdx.x + 1];
    const T block_start_ind = nnz_per_block * blockIdx.x;
    const T block_end_ind = min(nnz_per_block * (blockIdx.x + 1), nnz);
    const T block_num_nnz = block_end_ind - block_start_ind;
#pragma unroll
    for (int i = 0; i < R; i++)
    {
        const T shared_inx = threadIdx.x + i * blockDim.x;
        const T index = min(block_start_ind + shared_inx, nnz - 1);
        shared_data[shared_inx] = csr_val[index] * x[csr_col_ind[index]];
    }
    __syncthreads();
    if ((csr_row_ptr[block_end_row] % nnz_per_block) != 0 ||
        block_start_row == block_end_row)
    {
        block_end_row = min(block_end_row + 1, m);
    }
    if (REDUCE_OPTION == 0)
    {
        flat_reduce_direct2(
            alpha,
            csr_row_ptr,
            shared_data,
            y,
            block_start_row,
            block_end_row,
            block_start_ind,
            block_num_nnz);
    }
    else if (REDUCE_OPTION == 1)
    {
        flat_reduce_in_vector2<T, U, V, W, 4>(
            alpha,
            csr_row_ptr,
            shared_data,
            y,
            block_start_row,
            block_end_row,
            block_start_ind,
            block_num_nnz);
    }
    else if (REDUCE_OPTION == 2)
    {
        flat_reduce_in_vector_with_mem_coalescing2<T, U, V, W, BLOCK_SIZE, 4>(
            alpha,
            csr_row_ptr,
            shared_data,
            y,
            block_start_row,
            block_end_row,
            block_start_ind,
            block_num_nnz);
    }
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_acc(alphasparseHandle_t handle,
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
    constexpr int R = 2;
    constexpr T BLOCK_SIZE = 512;
    constexpr T nnz_per_block = R * BLOCK_SIZE;
    const T GRID_SIZE = ceildivT<T>(nnz, nnz_per_block);
    const T nwarps = GRID_SIZE;
    // printf("nwarps: %d\n", nwarps);
    const T GRID_SIZE2 = ceildivT<T>(nwarps + m, BLOCK_SIZE);
    // const T GRID_SIZE3 = ceildivT<T>(nwarps, BLOCK_SIZE);
    const T REDUCE_OPTION = 0;
    // printf("nwarps: %d\n", nwarps);
    // double time1 = get_time_us();
    // row partition
    T *partition = (T *)externalBuffer;
    hipEvent_t event_start, event_stop;
    float elapsed_time = 0.0;
    
    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    hipLaunchKernelGGL(balanced_partition_row_by_nnz_acc_and_scale_y, dim3(GRID_SIZE2), dim3(BLOCK_SIZE), 0, 0, 
        csr_row_ptr, m, nwarps, partition, nnz, nnz_per_block, beta, y);
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    printf("alphasparse pre: %f ms\n", elapsed_time);

    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(spmv_acc<T, U, V, W, BLOCK_SIZE, nnz_per_block, REDUCE_OPTION, R>), dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0, 
        m,
        nnz,
        alpha,
        partition,
        csr_row_ptr,
        csr_col_ind,
        csr_val,
        x,
        beta,
        y);
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    printf("alphasparse compute: %f ms\n", elapsed_time);
    // T *partition_host = (T *)malloc((nwarps + 1) * sizeof(T));
    // hipMemcpy(partition_host, partition, (nwarps + 1) * sizeof(T), hipMemcpyDeviceToHost);
    // for (int i = 0; i <= nwarps; i++)
    // {
    //     printf("%d, ", partition_host[i]);
    // }
    // printf("\n");
    // double time2 = get_time_us();
    // printf("pre time: %lf\n", (time2 - time1) / (1e3));

    // spmv
    // double time3 = get_time_us();
    // hipLaunchKernelGGL(HIP_KERNEL_NAME(csr_gemv_row_partition_parallel<BLOCK_SIZE, WFSIZE>), dim3(GRIDSIZE), dim3(BLOCK_SIZE), 0, handle->stream, m, alpha, partition, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    // double time4 = get_time_us();
    // printf("partition time: %lf\n", (time4 - time3) / (1e3));

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
