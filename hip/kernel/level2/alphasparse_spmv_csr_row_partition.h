#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_csr_vector.h"

template <typename T>
__device__ static T lower_bound_int(const T *t, T l, T r, T value)
{
    while (r > l)
    {
        T m = (l + r) / 2;
        if (t[m] < value)
        {
            l = m + 1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}

template <typename T>
__global__ static void balanced_partition_row_by_nnz(const T *acc_sum_arr, T rows, T num_wavefront, T *partition)
{
    const T idx = blockIdx.x * blockDim.x + threadIdx.x;
    const T stride = blockDim.x * gridDim.x;

    if (idx >= num_wavefront)
        return;

    const T nnz = acc_sum_arr[rows - 1];
    const T ave = nnz / num_wavefront;

    partition[idx] = lower_bound_int(acc_sum_arr, 0, rows - 1, (ave * idx));

    if (idx == 0)
    {
        partition[0] = 0;
        partition[num_wavefront] = rows;
    }
}

template <int BLOCK_SIZE, int WF_SIZE, typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCK_SIZE)
    __global__ static void csr_gemv_row_partition_vector(T m,
                                                         W alpha,
                                                         const T *partition,
                                                         const T *row_offset,
                                                         const T *csr_col_ind,
                                                         const U *csr_val,
                                                         const U *x,
                                                         W beta,
                                                         V *y)
{
    const T gid = blockIdx.x * BLOCK_SIZE + threadIdx.x; // global thread index
    const T lid = threadIdx.x & (WF_SIZE - 1);           // thread index within the wavefront
    const T wf_id = gid / WF_SIZE;                       // global wavefront index
    const T VECTORS_PER_BLOCK = BLOCK_SIZE / WF_SIZE;    // vector num in the block
    const T vector_lane = threadIdx.x / WF_SIZE;         // vector index within the block

    const T row_b = partition[wf_id];
    const T row_e = partition[wf_id + 1];

    // Loop over rows
    for (T row = row_b; row < row_e; row++)
    {
        T row_start, row_end;
        row_start = row_offset[row];
        row_end = row_offset[row + 1];

        V sum = {};

        // Loop over non-zero elements
        for (T j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
            // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
        }

        // Obtain row sum using parallel reduction
        // typedef hipcub::WarpReduce<ALPHA_Number, WF_SIZE> WarpReduce;
        //__shared__ typename WarpReduce::TempStorage temp_storage[VECTORS_PER_BLOCK];
        // sum = WarpReduce(temp_storage[vector_lane]).Sum(sum);
        sum = wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1)
        {
            y[row] = y[row] * beta + sum * alpha;
            // ALPHA_Number t1, t2;
            // alpha_mul(t1, y[row], beta);
            // alpha_mul(t2, sum, alpha);
            // alpha_add(y[row], t1, t2);
        }
    }
}

template <int BLOCK_SIZE, int WF_SIZE, typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCK_SIZE)
    __global__ static void csr_gemv_row_partition(T m,
                                                  W alpha,
                                                  const T *partition,
                                                  const T *row_offset,
                                                  const T *csr_col_ind,
                                                  const U *csr_val,
                                                  const U *x,
                                                  W beta,
                                                  V *y)
{
    const T gid = blockIdx.x * BLOCK_SIZE + threadIdx.x; // global thread index
    const T lid = threadIdx.x & (WF_SIZE - 1);           // thread index within the wavefront
    const T wf_id = gid / WF_SIZE;                       // global wavefront index
    const T VECTORS_PER_BLOCK = BLOCK_SIZE / WF_SIZE;    // vector num in the block
    const T vector_lane = threadIdx.x / WF_SIZE;         // vector index within the block

    const T row_b = partition[wf_id];
    const T row_e = partition[wf_id + 1];
    const T nnz = row_offset[row_e] - row_offset[row_b];
    const T rows = row_e - row_b;
    const T nnz_per_row = nnz / rows;
    const uint32_t sub_wf_size = max(min(1 << (31 - __clz(nnz_per_row)), WF_SIZE), 2);
    const T num_sub_wf = WF_SIZE / sub_wf_size;

    const T sub_lid = lid % sub_wf_size;
    const T sub_wid = lid / sub_wf_size;
    // bool head_flag = false;
    // typedef hipcub::WarpReduce<double, WF_SIZE> WarpReduce;
    //__shared__ typename WarpReduce::TempStorage temp_storage[VECTORS_PER_BLOCK];
    //  Loop over rows
    for (T idx = row_b; idx < row_e; idx += num_sub_wf)
    {
        const T row = idx + sub_wid;
        if (row >= row_e)
            continue;

        T row_start, row_end;
        row_start = row_offset[row];
        row_end = row_offset[row + 1];

        V sum = {};

        // Loop over non-zero elements
        for (T j = row_start + sub_lid; j < row_end; j += sub_wf_size)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
            // alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
        }
        // head_flag = !sub_lid;
        //  Obtain row sum using parallel reduction
        // sum = WarpReduce(temp_storage[vector_lane]).HeadSegmentedSum(sum, head_flag);
        sum = sub_wfreduce_sum(sum, sub_wf_size);
        // First thread of each wavefront writes result into global memory
        if (sub_lid == 0)
        {
            y[row] *= beta;
            // alpha_mule(y[row], beta);
            y[row] += sum * alpha;
            // alpha_madde(y[row], sum, alpha);
        }
    }
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_partition(alphasparseHandle_t handle,
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
    // const T CU        = 84;
    // const T SIMD      = 4;
    // const T OCCUPY    = 10;
    // const T ACTIVE_WF = CU * SIMD * OCCUPY;

    // const float RATIO = 0.5; //TODO: tune

    // const T BLOCK_SIZE     = 1024;
    // const T WFSIZE        = 32;
    // const T num_wavefront = ACTIVE_WF * RATIO;
    // const T GRIDSIZE      = num_wavefront * WFSIZE / BLOCK_SIZE;

    const T SM = 84;
    const T MAX_WARP_PER_SM = 64;

    const float RATIO = 0.5; // TODO: tune

    const T BLOCK_SIZE = 1024;
    const T WFSIZE = 32;
    const T num_wavefront = SM * MAX_WARP_PER_SM;
    const T GRIDSIZE = num_wavefront * WFSIZE / BLOCK_SIZE;

    printf("\n===========num_wavefront:%d, m:%d==============\n", num_wavefront, m);
    if (num_wavefront > m)
    {
        return spmv_csr_vector(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    }

    double time1 = get_time_us();
    // row partition
    T *partition;
    hipMalloc((void **)&partition, (num_wavefront + 1) * sizeof(T));
    hipLaunchKernelGGL(balanced_partition_row_by_nnz, dim3(GRIDSIZE), dim3(BLOCK_SIZE), 0, handle->stream, csr_row_ptr + 1, m, num_wavefront, partition);
    double time2 = get_time_us();
    printf("pre time: %lf\n", (time2 - time1) / (1e3));

    // spmv
    double time3 = get_time_us();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(csr_gemv_row_partition<BLOCK_SIZE, WFSIZE>), dim3(GRIDSIZE), dim3(BLOCK_SIZE), 0, handle->stream, m, alpha, partition, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    double time4 = get_time_us();
    printf("partition time: %lf\n", (time4 - time3) / (1e3));

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
