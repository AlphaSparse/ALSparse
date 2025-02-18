#pragma once

#include "alphasparse.h"
#include "hip/hip_runtime.h"
// #include <thrust/sort.h>
// #include <thrust/device_vector.h>
#include <rocprim/rocprim.hpp>

/*
./build/src/test/spsv_csr_r_f64_test_metrics --data-file=../matrix_test/2cubes_sphere.mtx --transA=N --fillA=L --diagA=N --iter=1 --warmup=0 --alg_num=9 --check --metrics
*/ 

template<unsigned int WARP_SIZE, typename U>
__device__ __forceinline__ static U
warp_reduce_max(
    const U num
) {
    U tmp_num = num;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        tmp_num = max(tmp_num, __shfl_xor(tmp_num, offset));
    }
    return tmp_num;
}

// __device__ __forceinline__ double 
// rocsparse_nontemporal_load(
//     const double* ptr
// ) { 
//     return __builtin_nontemporal_load(ptr); 
// }

template<unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void 
spsv_csr_n_lo_roc_analysis_kernel(
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
    T local_col = __ldg(&csr_col_idx[j]);
    // T local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
    while (j < row_end && local_col < first_row) {
        T local_done = done_array[local_col];
        local_max = max(local_done, local_max);
        j += (local_done != 0) * WARP_SIZE;
        if (local_done != 0 && j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
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
void
get_row_map_sorted(
    alphasparseHandle_t handle,
    const T m,
    T* done_array_i,
    T* done_array_o,
    T* row_map_i,
    T* row_map_o
) {
    unsigned int start_bit = 0;
    unsigned int end_bit = 64 - __builtin_clzll(m);
    void *temp_buffer = nullptr;
    size_t temp_size;
    rocprim::radix_sort_pairs(
        temp_buffer, 
        temp_size, 
        done_array_i, 
        done_array_o,
        row_map_i,
        row_map_o,
        m,
        start_bit,
        end_bit,
        handle->stream
    );
    hipMalloc(&temp_buffer, temp_size);
    rocprim::radix_sort_pairs(
        temp_buffer, 
        temp_size, 
        done_array_i, 
        done_array_o,
        row_map_i,
        row_map_o,
        m,
        start_bit,
        end_bit,
        handle->stream
    );
    hipFree(temp_buffer);
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_roc_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    T* dpd_col,
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
        (spsv_csr_n_lo_roc_analysis_kernel<BLOCKSIZE, WARP_SIZE>), 
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
    T *done_array_o;
    hipMalloc((void **)&done_array_o, m * sizeof(T));
    get_row_map_sorted(handle, m, done_array, done_array_o, row_map, row_map);
    get_row_map_sorted(handle, m, done_array, done_array, dpd_col, dpd_col);
    hipFree(done_array_o);
    hipDeviceSynchronize();
    T *h_done_array = (T *)malloc(m * sizeof(T));
    T *h_row_map = (T *)malloc(m * sizeof(T));
    T *h_dpd_col = (T *)malloc(m * sizeof(T));
    hipMemcpy(h_done_array, done_array, m * sizeof(T), hipMemcpyDeviceToHost);
    hipMemcpy(h_row_map, row_map, m * sizeof(T), hipMemcpyDeviceToHost);
    hipMemcpy(h_dpd_col, dpd_col, m * sizeof(T), hipMemcpyDeviceToHost);
    // for (int i = 0; i < m; i++) {
    //     printf("idx: %d - done: %d - row: %d - col: %d\n", i, h_done_array[i], h_row_map[i], h_dpd_col[i]);
    // }
    free(h_done_array);
    free(h_row_map);
    free(h_dpd_col);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<unsigned int WARP_SIZE, typename U>
__device__ __forceinline__ static U
warp_reduce_sum(
    U *num
) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        *num += __shfl_up_sync(0xFFFFFFFF, *num, offset);
    }
    return *num;
}

template <bool SLEEP>
__device__ __forceinline__ int 
spin_loop(
    int* __restrict__ done, 
    int scope
) {
    int local_done = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
    uint32_t times_through = 0;
    while(!local_done) {
        if(SLEEP) {
            for(uint32_t i = 0; i < times_through; ++i) {
                __builtin_amdgcn_s_sleep(1);
            }
            if(times_through < 3907) {
                ++times_through;
            }
        }
        local_done = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
    }
    return local_done;
}

template<unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc_solve_kernel(
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
    int lid = hipThreadIdx_x & (WF_SIZE - 1);
    int wid = hipThreadIdx_x / WF_SIZE;
    // Scalarize wid, i.e. move it from a vector register to a scalar register, so all dependent
    // values can be loaded or computed with scalar instructions (idx, row, row_begin...)
    wid = __builtin_amdgcn_readfirstlane(wid);
    // Index into the row map
    T idx = hipBlockIdx_x * (BLOCKSIZE / WF_SIZE) + wid;
    // Shared memory to hold diagonal entry
    __shared__ double diagonal[BLOCKSIZE / WF_SIZE];
    // Do not run out of bounds
    if(idx >= m) {
        return;
    }
    // Get the row this warp will operate on
    T row = row_map[idx];
    // Current row entry point and exit point
    T row_begin = csr_row_ptr[row];
    T row_end   = csr_row_ptr[row + 1];
    // Local summation variable.
    double local_sum = static_cast<double>(0);
    if(lid == 0) {
        // Lane 0 initializes its local sum with alpha and x
        local_sum = alpha * __builtin_nontemporal_load(&x[row]);
    }
    for(T j = row_begin + lid; j < row_end; j += WF_SIZE) {
        // Current column this lane operates on
        T local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
        // Local value this lane operates with
        double local_val = __builtin_nontemporal_load(&csr_val[j]);
        // Differentiate upper and lower triangular mode
        // if(fill_mode == rocsparse_fill_mode_upper) {
        //     // Processing upper triangular
        //     // Ignore all entries that are below the diagonal
        //     if(local_col < row) {
        //         continue;
        //     }
        //     // Diagonal entry
        //     if(local_col == row) {
        //         // If diagonal type is non unit, do division by diagonal entry
        //         // This is not required for unit diagonal for obvious reasons
        //         // if(diag_type == rocsparse_diag_type_non_unit) {
        //             diagonal[wid] = static_cast<double>(1) / local_val;
        //         // }
        //         continue;
        //     }
        // }
        // else 
        // if(fill_mode == rocsparse_fill_mode_lower) {
            // Processing lower triangular
            // Ignore all entries that are above the diagonal
            if(local_col > row) {
                break;
            }
            // Diagonal entry
            if(local_col == row) {
                // If diagonal type is non unit, do division by diagonal entry
                // This is not required for unit diagonal for obvious reasons
                // if(diag_type == rocsparse_diag_type_non_unit) {
                    diagonal[wid] = static_cast<double>(1) / local_val;
                // }
                break;
            }
        // }
        // Spin loop until dependency has been resolved
        (void)spin_loop<true>((int *)&done_array[local_col], (int)__HIP_MEMORY_SCOPE_AGENT);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        // Local sum computation for each lane
        // __threadfence();
        local_sum = std::fma(-local_val, y[local_col], local_sum);
    }
    // Gather all local sums for each lane
    local_sum = wfreduce_sum<WF_SIZE>(local_sum);
    // If we have non unit diagonal, take the diagonal into account
    // For unit diagonal, this would be multiplication with one
    // if(diag_type == rocsparse_diag_type_non_unit) {
        // __threadfence_block();
        local_sum = local_sum * diagonal[wid];
    // }
    if(lid == WF_SIZE - 1) {
        // Store the rows result in y
        __builtin_nontemporal_store(local_sum, &y[row]);
        // Mark row as done
        __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc_solve_kernel(
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
spsv_csr_n_lo_roc_solve_kernel(
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
spsv_csr_n_lo_roc_solve_kernel(
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
spsv_csr_n_lo_roc_solve(
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
    hipLaunchKernelGGL(
        (spsv_csr_n_lo_roc_solve_kernel<BLOCKSIZE, WARP_SIZE>),
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