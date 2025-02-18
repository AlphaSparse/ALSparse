#pragma once

#include "alphasparse.h"
#include "hip/hip_runtime.h"
#include <rocprim/rocprim.hpp>

// level-scheduling
// single block per level

// ./build/src/test/spsv_csr_r_f64_test_metrics --data-file=../matrix_test/2cubes_sphere.mtx --transA=N --fillA=L --diagA=N --iter=1 --warmup=0 --alg_num=11 --check --metrics


template<unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void 
spsv_csr_n_lo_sbpl_analysis_kernel(
    const T m,
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    volatile T* __restrict__ done_array,
    T* __restrict__ row_map
){
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T first_row = blockIdx.x * (blockDim.x / WARP_SIZE);
    T row = first_row + wid;
    volatile __shared__ T local_done_array[BLOCKSIZE / WARP_SIZE];
    if (lid == 0) {
        local_done_array[wid] = 0;
    }
    __syncthreads();
    if (row >= m) {
        return;
    }
    if (lid == 0) {
        row_map[row] = row;
    }
    T local_max = 0;
    T row_begin = csr_row_ptr[row];
    T row_end = csr_row_ptr[row + 1];
    T j = row_begin + lid;
    T local_col = csr_col_idx[j];
    // T local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
    while (j < row_end && local_col < first_row) {
        T local_done = done_array[local_col];
        local_max = max(local_done, local_max);
        j += (local_done != 0) * WARP_SIZE;
        if (local_done != 0 && j < row_end) {
            local_col = csr_col_idx[j];
        }
    }
    while (j < row_end && local_col < row) {
        T local_idx = local_col - first_row;
        T local_done = local_done_array[local_idx];
        j += (local_done != 0) * WARP_SIZE;
        local_max = max(local_done, local_max);
    }
    local_max = warp_reduce_max<WARP_SIZE>(local_max);
    if (lid == WARP_SIZE - 1) {
        local_done_array[wid] = local_max + 1;
        done_array[row] = local_max + 1;
    }
    return;
}

template<typename T>
__global__ static void 
get_level_ptr(
    const T* __restrict__ done_array,
    const T m,
    T* __restrict__ level_ptr,
    T* level_size
) {
    const T tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) {
        return;
    }
    T cur_degree = done_array[tid];
    if (tid == 0 || cur_degree != done_array[tid - 1]) {
        level_ptr[cur_degree - 1] = tid;
    }
    if (tid == m - 1) {
        level_ptr[cur_degree] = m;
        *level_size = cur_degree + 1;
    }
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_sbpl_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    T* level_ptr,
    T* level_size,
    const U* x,
    U* y,
    void* externalBuffer
) {
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 64;  // 设置为32 - 运行时死锁？
    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid = dim3((m - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
    T *done_array = reinterpret_cast<T*>(externalBuffer);
    hipMemset(done_array, 0, m * sizeof(T));
    hipLaunchKernelGGL(
        (spsv_csr_n_lo_sbpl_analysis_kernel<BLOCKSIZE, WARP_SIZE>), 
        blockPerGrid, 
        threadPerBlock, 
        0, 
        handle->stream, 
        m,
        csr_row_ptr,
        csr_col_idx,
        done_array,
        row_map
    );
    // T *done_array_o;
    // hipMalloc((void **)&done_array_o, m * sizeof(T));
    get_row_map_sorted(handle, m, done_array, done_array, row_map, row_map);
    // hipFree(done_array_o);

    T *d_level_size;
    hipMalloc((void **)&d_level_size, sizeof(T));

    blockPerGrid = dim3((m - 1) / threadPerBlock.x + 1);
    hipLaunchKernelGGL(
        get_level_ptr,
        blockPerGrid, 
        threadPerBlock, 
        0, 
        handle->stream, 
        done_array,
        m,
        level_ptr,
        d_level_size
    );

    hipMemcpy(level_size, d_level_size, sizeof(T), hipMemcpyDeviceToHost);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_sbpl_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_size
) {
    const unsigned int SEGM_SIZE = BLOCKSIZE / WARP_SIZE;   // 1 block每次处理SEGM_SIZE个row
    T lid = threadIdx.x & (WARP_SIZE - 1);                  // lane_id
    T wid = threadIdx.x / WARP_SIZE;                        // local_row_id
    volatile __shared__ double diag[BLOCKSIZE / WARP_SIZE];
    for (int level_id = 0; level_id < level_size; level_id++) {  // per level
        for (int row_idx = level_ptr[level_id]; row_idx < level_ptr[level_id + 1]; row_idx += SEGM_SIZE) {  // per row
            T row_id = row_map[row_idx + wid];      // 当前WARP需要计算的row 
            T row_begin = csr_row_ptr[row_id];
            T row_end = csr_row_ptr[row_id + 1];
            double local_sum = {};
            if (lid == 0) {
                local_sum = alpha * x[row_id];
            }
            T val_id = row_begin + lid;
            T col_id = m;
            double val = {};
            if (val_id < row_end) {
                col_id = csr_col_idx[val_id];
                val = csr_val[val_id];
            }
            while (val_id < row_end && col_id < row_id) { // 当前线程需要处理的nnz
                local_sum -= val * y[col_id];
                val_id += WARP_SIZE;
                if (val_id < row_end) {
                    col_id = csr_col_idx[val_id];
                    val = csr_val[val_id];
                }
            }
            if (col_id == row_id) {
                diag[wid] = double(1) / val;
            }
            warp_reduce_sum<WARP_SIZE>(&local_sum);
            if (lid == WARP_SIZE - 1) {
                y[row_id] = local_sum * diag[wid];
            }
        }
        __syncthreads();
    }
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_sbpl_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const float* __restrict__ csr_val,
    const T m,
    const T nnz,
    const float alpha,
    const float* __restrict__ x,
    float* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_size
) {
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_sbpl_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const hipDoubleComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const hipDoubleComplex alpha,
    const hipDoubleComplex* __restrict__ x,
    hipDoubleComplex* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_size
) {
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_sbpl_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const hipFloatComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const hipFloatComplex alpha,
    const hipFloatComplex* __restrict__ x,
    hipFloatComplex* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_size
) {
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_sbpl_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const T* row_map,
    const T* level_ptr,
    const T level_size,
    const U* x,
    U* y,
    void *externalBuffer
) {
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 64;

    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid= dim3(1);

    T *done_array = reinterpret_cast<T*>(externalBuffer);
    hipMemset(done_array, 0, m * sizeof(T));

    hipLaunchKernelGGL(
        (spsv_csr_n_lo_sbpl_solve_kernel<BLOCKSIZE, WARP_SIZE>),
        blockPerGrid, 
        threadPerBlock, 
        0, 
        handle->stream,
        csr_row_ptr,
        csr_col_idx,
        csr_val,
        m,
        nnz,
        alpha,
        x,
        y,
        row_map,
        level_ptr,
        level_size
    );
    return ALPHA_SPARSE_STATUS_SUCCESS;
}