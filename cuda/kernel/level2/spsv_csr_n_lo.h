#pragma once

#include "alphasparse.h"
#include "alphasparse/types.h" 

template<typename T, typename U>
__global__ static void
spsv_csr_n_lo_kernel(const T* csrRowPtr,
                     const T* csrColIdx,
                     const U* csrVal,
                     volatile T* get_value,
                     const T m,
                     const T nnz,
                     const U alpha,
                     const U* x,
                     U* y,
                     U* diag)
{
  const T global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const T stride = blockDim.x * gridDim.x;

  // TODO:opt here
  // diag = csrVal[csrRotPtr[row+1]-1];
  for (T r = global_id; r < m; r += stride) {
    for (T ai = csrRowPtr[r]; ai < csrRowPtr[r + 1]; ai++) {
      T ac = csrColIdx[ai];
      if (ac == r) {
        diag[r] = csrVal[ai];
      }
    }
  }
  T col, j;
  U yi;
  U left_sum = {};

  for (T i = global_id; i < m; i += stride) {
    left_sum = {};
    j = csrRowPtr[i];
    while (j < csrRowPtr[i + 1]) {
      col = csrColIdx[j];
      while (get_value[col] == 1) {
        if (col < i) {
          left_sum += csrVal[j] * y[col];
        } else
          break;
        j++;
        col = csrColIdx[j];
      }
      __syncthreads();
      T tmp_try = !(i ^ col);
      U tmp = alpha * x[i];
      tmp = tmp - left_sum;
      yi = tmp / diag[i];
      y[i] = tmp_try * yi + (1 - tmp_try) * y[i];
      __syncthreads();
      get_value[i] = tmp_try | get_value[i];
      __threadfence();
      if (tmp_try)
        break;
    }
  }
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo(alphasparseHandle_t handle,
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
  U* diag;
  cudaMalloc((void**)&diag, sizeof(U) * m);
  cudaMemset(diag, '\0', sizeof(U) * m);

  T* get_value;
  cudaMalloc((void**)&get_value, (m) * sizeof(T));
  cudaMemset(get_value, 0, sizeof(T) * m);
  spsv_csr_n_lo_kernel<T, U>
    <<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(
      csr_row_ptr, csr_col_ind, csr_val, get_value, m, nnz, alpha, x, y, diag);
  cudaFree(get_value);
  cudaFree(diag);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
