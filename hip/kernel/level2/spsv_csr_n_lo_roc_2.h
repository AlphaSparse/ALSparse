#pragma once

#include "alphasparse.h"
#include "hip/hip_runtime.h"
#include <rocprim/rocprim.hpp>

/*
./build/src/test/spsv_csr_r_f64_test_metrics --data-file=../matrix_test/2cubes_sphere.mtx --transA=N --fillA=L --diagA=N --iter=1 --warmup=0 --alg_num=10 --check --metrics
*/

// template<unsigned int WARP_SIZE, typename U>
// __device__ __forceinline__ static U
// warp_reduce_max(
//     U *num
// ) {
//     for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
//         *num = max(*num, __shfl_xor_sync(0xFFFFFFFF, *num, offset));
//     }
//     return *num;
// }

// __device__ __forceinline__ double 
// rocsparse_nontemporal_load(
//     const double* ptr
// ) { 
//     return __builtin_nontemporal_load(ptr); 
// }

template<unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void 
spsv_csr_n_lo_roc2_analysis_kernel(
    const T m,
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    T* __restrict__ done_array,
    T* __restrict__ row_map
){
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T first_row = blockIdx.x * (blockDim.x / WARP_SIZE);
    T row = first_row + wid;
    __shared__ T local_done_array[BLOCKSIZE / WARP_SIZE];
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
    // T local_col = __ldg(&csr_col_idx[j]);
    T local_col = csr_col_idx[j];
    // T local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
    while (j < row_end && local_col < first_row) {
        __threadfence();
        T local_done = done_array[local_col];
        // T local_done = __hip_atomic_load(&done_array[local_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        local_max = max(local_done, local_max);
        j += (local_done != 0) * WARP_SIZE;
        if (local_done != 0 && j < row_end) {
            // local_col =  __ldg(&csr_col_idx[j]);
            local_col = csr_col_idx[j];
            // local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
        }
    }
    while (j < row_end && local_col < row) {
        T local_idx = local_col - first_row;
        __threadfence_block();
        T local_done = local_done_array[local_idx];
        // T local_done = __hip_atomic_load(&local_done_array[local_idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        j += (local_done != 0) * WARP_SIZE;
        local_max = max(local_done, local_max);
    }
    local_max = warp_reduce_max<WARP_SIZE>(local_max);
    if (lid == WARP_SIZE - 1) {
        // local_done_array[wid] = local_max + 1;
        // __threadfence_block();
        __hip_atomic_store(&local_done_array[wid], local_max + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        done_array[row] = local_max + 1;
        __threadfence();
        // __hip_atomic_store(&done_array[row], local_max + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return;
}

// template<typename T>
// void
// get_row_map_sorted(
//     alphasparseHandle_t handle,
//     const T m,
//     T* done_array,
//     T* row_map
// ) {
//     unsigned int start_bit = 0;
//     unsigned int end_bit = 64 - __builtin_clzll(m);
//     void *temp_buffer = nullptr;
//     size_t temp_size;
//     rocprim::radix_sort_pairs(
//         temp_buffer, 
//         temp_size, 
//         done_array, 
//         done_array,
//         row_map,
//         row_map,
//         m,
//         start_bit,
//         end_bit,
//         handle->stream
//     );
//     hipMalloc(&temp_buffer, temp_size);
//     rocprim::radix_sort_pairs(
//         temp_buffer, 
//         temp_size, 
//         done_array, 
//         done_array,
//         row_map,
//         row_map,
//         m,
//         start_bit,
//         end_bit,
//         handle->stream
//     );
//     hipFree(temp_buffer);
//     return;
// }

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_roc2_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    const U* x,
    U* y,
    void* externalBuffer
) {
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 64;  // 设置为32 - 运行时死锁？
    const dim3 threadPerBlock = dim3(BLOCKSIZE);
    const dim3 blockPerGrid = dim3((m - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
    T *done_array = reinterpret_cast<T*>(externalBuffer);
    hipMemset(done_array, 0, m * sizeof(T));
    hipLaunchKernelGGL(
        (spsv_csr_n_lo_roc2_analysis_kernel<BLOCKSIZE, WARP_SIZE>), 
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
    // get_row_map_sorted(handle, m, done_array, row_map);
    get_row_map_sorted(handle, m, done_array, done_array, row_map, row_map);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

// template<unsigned int WARP_SIZE, typename U>
// __device__ __forceinline__ static U
// warp_reduce_sum(
//     U *num
// ) {
//     for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
//         *num += __shfl_up_sync(0xFFFFFFFF, *num, offset);
//     }
//     return *num;
// }

// template<typename U>
// __device__ __forceinline__ static U
// my_fma(
//     U a,
//     U b,
//     U c
// ) {
//     return fma(a, b, c);
// }

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc2_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    double* __restrict__ y,
    const T* __restrict__ row_map,
    T* __restrict__ done_array
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T idx = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    __shared__ double diag[BLOCKSIZE / WARP_SIZE];
    if (idx >= m) {
        return;
    }
    T row = row_map[idx];
    T row_begin = csr_row_ptr[row];
    T row_end = csr_row_ptr[row + 1];
    double local_sum = {};
    if (lid == 0) {
        local_sum = alpha * x[row];
        // local_sum = alpha * __ldg(&x[row]);
        // local_sum = alpha * __builtin_nontemporal_load(&x[row]);
    }
    T j = row_begin + lid;
    T local_col = m;
    double local_val = {};
    if (j < row_end) {
        local_col = csr_col_idx[j];
        // local_col = __ldg(&csr_col_idx[j]);
        // local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
        local_val = csr_val[j];
        // local_val = __ldg(&csr_val[j]);
        // local_val = __builtin_nontemporal_load(&csr_val[j]);
    } 
    while (j < row_end && local_col < row) {
        __threadfence();
        int t = (done_array[local_col] != 0);
        // int t = __hip_atomic_load(&done_array[local_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);;
        j += t * WARP_SIZE;
        if (t) {
            // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            // __threadfence();
            // local_sum = my_fma(-local_val, y[local_col], local_sum);
            local_sum = std::fma(-local_val, y[local_col], local_sum);
        }
        if (t && j < row_end) {
            local_col = csr_col_idx[j];
            // local_col = __ldg(&csr_col_idx[j]);
            // local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
            local_val = csr_val[j];
            // local_val = __ldg(&csr_val[j]);
            // local_val = __builtin_nontemporal_load(&csr_val[j]);
        }
    }
    if (local_col == row) {
        diag[wid] = double(1) / local_val;
    }
    warp_reduce_sum<WARP_SIZE>(&local_sum);
    if (lid == WARP_SIZE - 1) {
        y[row] = local_sum * diag[wid];
        // __builtin_nontemporal_store(local_sum * diag[wid], &y[row]);
        __threadfence();
        done_array[row] = 1;
        // __threadfence(); 
        // __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc2_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const float* __restrict__ csr_val,
    const T m,
    const T nnz,
    const float alpha,
    const float* __restrict__ x,
    float* __restrict__ y,
    const T* __restrict__ row_map,
    T* __restrict__ done_array
) {
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc2_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const hipDoubleComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const hipDoubleComplex alpha,
    const hipDoubleComplex* __restrict__ x,
    hipDoubleComplex* __restrict__ y,
    const T* __restrict__ row_map,
    T* __restrict__ done_array
) {
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc2_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const hipFloatComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const hipFloatComplex alpha,
    const hipFloatComplex* __restrict__ x,
    hipFloatComplex* __restrict__ y,
    const T* __restrict__ row_map,
    T* __restrict__ done_array
) {
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_roc2_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    const U* x,
    U* y,
    void *externalBuffer
) {
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 64;

    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid= dim3((m - 1) / (threadPerBlock.x / WARP_SIZE) + 1);

    T *done_array = reinterpret_cast<T*>(externalBuffer);
    hipMemset(done_array, 0, m * sizeof(T));

    // spsv_csr_n_lo_roc2_solve_kernel<BLOCKSIZE, WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
    //     csr_row_ptr,
    //     csr_col_idx,
    //     csr_val,
    //     m,
    //     nnz,
    //     alpha,
    //     x,
    //     y,
    //     row_map,
    //     done_array
    // );
    hipLaunchKernelGGL(
        (spsv_csr_n_lo_roc2_solve_kernel<BLOCKSIZE, WARP_SIZE>),
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
        done_array
    );


    return ALPHA_SPARSE_STATUS_SUCCESS;
}