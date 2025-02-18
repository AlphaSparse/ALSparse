#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"

template<typename T, typename U>
__global__ static void
spsv_csr_u_lo_kernel(const T* csrRowPtr,
                     const T* csrColIdx,
                     const U* csrVal,
                     volatile T* get_value,
                     const T m,
                     const T nnz,
                     const U alpha,
                     const U* x,
                     U* y)
{
  const T global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const T stride = blockDim.x * gridDim.x;

  T row_end, col_end;
  T col, j;
  U yi;
  U left_sum = {};

  for (T i = global_id; i < m; i += stride) {
    row_end = csrRowPtr[i + 1];
    left_sum = {};
    j = csrRowPtr[i];

    col_end = csrColIdx[row_end - 1];

    y[i] = alpha; // init yi for nnz_row==0 case

    while (j < csrRowPtr[i + 1]) {
      col = csrColIdx[j];
      bool flag = false; // 记录是否到达对角线，当csr存储的对角线没有元素时

      while (get_value[col] == 1) {
        if (col < i && col <= col_end) {
          left_sum += csrVal[j] * y[col];
        }

        if (col >= col_end) {
          flag = true;
          break;
        }
        j++;
        col = csrColIdx[j];
      }
      __syncthreads();
      T tmp_try = (!(i ^ col)) | (flag) | (col > i);
      yi = alpha * x[i] - left_sum;
      y[i] = tmp_try * yi + (1 - tmp_try) * y[i];
      __syncthreads();
      get_value[i] = tmp_try | get_value[i];
      __threadfence();

      if (tmp_try)
        break;
    }
    get_value[i] = 1; // tag get_value for nnz_row==0 case
    __threadfence();
  }
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_u_lo(alphasparseHandle_t handle,
              T m,
              T nnz,
              const U alpha,
              const U* csr_val,
              const T* csr_row_ptr,
              const T* csr_col_ind,
              const U* x,
              U* y)
{
  const int threadPerBlock = 256;
  const int blockPerGrid = (m - 1) / threadPerBlock + 1;

  T* get_value;
  hipMalloc((void**)&get_value, (m) * sizeof(T));
  hipMemset(get_value, 0, sizeof(T) * m);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(spsv_csr_u_lo_kernel<T, U>), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, 
      csr_row_ptr, csr_col_ind, csr_val, get_value, m, nnz, alpha, x, y);
  hipFree(get_value);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
