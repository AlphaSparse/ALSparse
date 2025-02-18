#pragma once

#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

__device__ static int32_t lower_bound_int(const ALPHA_INT *t, ALPHA_INT l, ALPHA_INT r, ALPHA_INT value)
{
    while (r > l) {
        ALPHA_INT m = (l + r) / 2;
        if (t[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__global__ static void balanced_partition_row_by_nnz(const ALPHA_INT *acc_sum_arr, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition)
{
    const ALPHA_INT idx    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    if (idx >= num_threads) return;

    const ALPHA_INT nnz = acc_sum_arr[rows - 1];
    const ALPHA_INT ave = nnz / num_threads;

    partition[idx] = lower_bound_int(acc_sum_arr, 0, rows - 1, (ave * idx));

    if (idx == 0) {
        partition[0]           = 0;
        partition[num_threads] = rows;
    }
}

template <ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE>
__launch_bounds__(BLOCKSIZE)
    __global__ static void csr_gemv_row_partition_vector(ALPHA_INT m,
                                                         ALPHA_Number alpha,
                                                         const ALPHA_INT *partition,
                                                         const ALPHA_INT *row_offset,
                                                         const ALPHA_INT *csr_col_ind,
                                                         const ALPHA_Number *csr_val,
                                                         const ALPHA_Number *x,
                                                         ALPHA_Number beta,
                                                         ALPHA_Number *y)
{
    const ALPHA_INT gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x; // global thread index
    const ALPHA_INT lid   = hipThreadIdx_x & (WF_SIZE - 1); // thread index within the wavefront
    const ALPHA_INT wf_id = gid / WF_SIZE; // global wavefront index

    const ALPHA_INT row_b = partition[wf_id];
    const ALPHA_INT row_e = partition[wf_id + 1];

    // Loop over rows
    for (ALPHA_INT row = row_b; row < row_e; row++) {
        ALPHA_INT row_start, row_end;
        row_start = row_offset[row];
        row_end   = row_offset[row + 1];

        ALPHA_Number sum;
        alpha_setzero(sum);

        // Loop over non-zero elements
        for (ALPHA_INT j = row_start + lid; j < row_end; j += WF_SIZE) {
            // sum += alpha * csr_val[j] * x[csr_col_ind[j]];
            alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
        }

        // Obtain row sum using parallel reduction
        sum = wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if (lid == WF_SIZE - 1) {
            ALPHA_Number t1, t2;
            alpha_mul(t1, y[row], beta);
            alpha_mul(t2, sum, alpha);
            alpha_add(y[row], t1, t2);
        }
    }
}

template <ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE>
__launch_bounds__(BLOCKSIZE)
    __global__ static void csr_gemv_row_partition(ALPHA_INT m,
                                                  ALPHA_Number alpha,
                                                  const ALPHA_INT *partition,
                                                  const ALPHA_INT *row_offset,
                                                  const ALPHA_INT *csr_col_ind,
                                                  const ALPHA_Number *csr_val,
                                                  const ALPHA_Number *x,
                                                  ALPHA_Number beta,
                                                  ALPHA_Number *y)
{
    const ALPHA_INT gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x; // global thread index
    const ALPHA_INT lid   = hipThreadIdx_x & (WF_SIZE - 1); // thread index within the wavefront
    const ALPHA_INT wf_id = gid / WF_SIZE; // global wavefront index

    const ALPHA_INT row_b       = partition[wf_id];
    const ALPHA_INT row_e       = partition[wf_id + 1];
    const ALPHA_INT nnz         = row_offset[row_e] - row_offset[row_b];
    const ALPHA_INT rows        = row_e - row_b;
    const ALPHA_INT nnz_per_row = nnz / rows;
    const uint32_t sub_wf_size  = max(min(1 << (31 - __clz(nnz_per_row)), WF_SIZE), 2);
    const ALPHA_INT num_sub_wf  = WF_SIZE / sub_wf_size;

    const ALPHA_INT sub_lid = lid % sub_wf_size;
    const ALPHA_INT sub_wid = lid / sub_wf_size;

    // Loop over rows
    for (ALPHA_INT idx = row_b; idx < row_e; idx += num_sub_wf) {
        const ALPHA_INT row = idx + sub_wid;
        if (row >= row_e) continue;

        ALPHA_INT row_start, row_end;
        row_start = row_offset[row];
        row_end   = row_offset[row + 1];

        ALPHA_Number sum = ALPHA_ZERO;

        // Loop over non-zero elements
        for (ALPHA_INT j = row_start + sub_lid; j < row_end; j += sub_wf_size) {
            alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
        }

        // Obtain row sum using parallel reduction
        sum = sub_wfreduce_sum(sum, sub_wf_size);

        // First thread of each wavefront writes result into global memory
        if (sub_lid == sub_wf_size - 1) {
            alpha_mule(y[row], beta);
            alpha_madde(y[row], sum, alpha);
        }
    }
}

alphasparse_status_t csr_gemv_row_partition_dispatch(alphasparseHandle_t handle,
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
                                                     u_int32_t flag,
                                                     alphasparse_dcu_mat_info_t info)
{
    const ALPHA_INT CU        = 64;
    const ALPHA_INT SIMD      = 4;
    const ALPHA_INT OCCUPY    = 10;
    const ALPHA_INT ACTIVE_WF = CU * SIMD * OCCUPY;

    const float RATIO = 0.5; //TODO: tune

    const ALPHA_INT BLOCKSIZE     = 256;
    const ALPHA_INT WFSIZE        = 64;
    const ALPHA_INT num_wavefront = ACTIVE_WF * RATIO;
    const ALPHA_INT GRIDSIZE      = num_wavefront * WFSIZE / BLOCKSIZE;
    
    if (num_wavefront > m) {
        return csr_gemv_vector_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    }

    if (info == nullptr || info->csrmv_info == nullptr) {
        printf("null info or null csrmv_info.\n");
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    alphasparse_dcu_csrmv_info_t csrmv_info = info->csrmv_info;

    if (!csrmv_info->csr_rowpartition_has_tuned) {
        double time = get_time_us();
        // row partition
        ALPHA_INT *partition;
        hipMalloc((void **)&partition, (num_wavefront + 1) * sizeof(ALPHA_INT));
        hipLaunchKernelGGL((balanced_partition_row_by_nnz), dim3(num_wavefront / BLOCKSIZE), dim3(BLOCKSIZE), 0, handle->stream, csr_row_ptr + 1, m, num_wavefront, partition);
        time = (get_time_us() - time) / (1e3);
        // printf("partition time: %lf\n", time);

        csrmv_info->partition                  = partition;
        csrmv_info->csr_rowpartition_has_tuned = true;
    }

    if (csrmv_info->partition == nullptr) {
        printf("null partition point.\n");
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // spmv
    hipLaunchKernelGGL((csr_gemv_row_partition<BLOCKSIZE, WFSIZE>),
                       dim3(GRIDSIZE),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m,
                       alpha,
                       csrmv_info->partition,
                       csr_row_ptr,
                       csr_col_ind,
                       csr_val,
                       x,
                       beta,
                       y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}