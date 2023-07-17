#pragma once

#include "alphasparse.h"

template<typename T, typename U>
__global__ static void
spsv_csr_u_up_kernel(const T* csrRowPtr,
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

  T col, j, repeat = 0;
  U yi;
  U left_sum = {};

  for (T i = global_id; i < m; i += stride) {
    left_sum = {};
    T ii = m - 1 - i;
    j = csrRowPtr[m - i] - 1;

    const T row_start = csrRowPtr[i];
    const T row_end = csrRowPtr[i + 1];
    const T nnz_per_row = row_end - row_start;
    const T col_end = csrColIdx[row_end - 1];

    if (nnz_per_row == 0 || col_end < i) {
      y[i] = alpha;
      get_value[i] = 1;
      __threadfence();
      continue;
    }

    while (j >= csrRowPtr[ii]) {
      col = csrColIdx[j];

      while (get_value[col] == 1) {
        if (col > ii) {
          left_sum += csrVal[j] * y[col];
        } else
          break;
        j--;
        col = csrColIdx[j];
      }
      __syncthreads();
      T tmp_try = !(ii ^ col);
      yi = alpha * x[ii] - left_sum;
      y[ii] = tmp_try * yi + (1 - tmp_try) * y[ii];
      __syncthreads();
      get_value[m - 1 - i] = tmp_try | get_value[m - 1 - i];
      __threadfence();

      if (tmp_try)
        break;
    }
  }
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_u_up(alphasparseHandle_t handle,
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
  cudaMalloc((void**)&get_value, (m) * sizeof(T));
  cudaMemset(get_value, 0, sizeof(T) * m);
  spsv_csr_u_up_kernel<T, U>
    <<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(
      csr_row_ptr, csr_col_ind, csr_val, get_value, m, nnz, alpha, x, y);
  cudaFree(get_value);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
