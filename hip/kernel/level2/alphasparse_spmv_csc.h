#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse_spmv_coo.h"

template<typename T, typename U, typename V, typename W>
__global__ static void
spmv_csc_kernel(T m,
                T n,
                T nnz,
                const W alpha,
                const U* csc_val,
                const T* csc_col_ptr,
                const T* csc_row_ind,
                const U* x,
                const W beta,
                V* y)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (T i = ix; i < n; i += stride) {
    for (T ai = csc_col_ptr[i]; ai < csc_col_ptr[i + 1]; ai++) {
        V tmp = alpha * csc_val[ai] * x[i];
        atomicAdd(&y[csc_row_ind[ai]], tmp);
    }
  }
}

template<typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmv_csc(alphasparseHandle_t handle,
         T m,
         T n,
         T nnz,
         const W alpha,
         const U* csc_val,
         const T* csc_row_ind,
         const T* csc_col_ptr,
         const U* x,
         const W beta,
         V* y)
{
  const int threadPerBlock = 1024;
  const int blockPerGrid = (m - 1) / threadPerBlock + 1;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mulbeta<T, W, V>), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, beta, y);
  printf("\nscalar is runing!\n");
  hipLaunchKernelGGL(spmv_csc_kernel, dim3(1), dim3(1), 0, handle->stream, 
    m, n, nnz, alpha, csc_val, csc_col_ptr, csc_row_ind, x, beta, y);
  
  return ALPHA_SPARSE_STATUS_SUCCESS;
}