#pragma once

#include "alphasparse.h"

template<typename T, typename U>
__global__ static void
get_diags(T m,
          const U* csr_val,
          const T* csr_row_ptr,
          const T* csr_col_ind,
          U* diag)
{
  const T global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const T stride = blockDim.x * gridDim.x;
  for (T r = global_id; r < m; r += stride) {
    for (T ai = csr_row_ptr[r]; ai < csr_row_ptr[r + 1]; ai++) {
      T ac = csr_col_ind[ai];
      if (ac == r) {
        diag[r] = csr_val[ai];
      }
    }
  }
}

template<unsigned int DIM_X, unsigned int DIM_Y, typename T, typename U>
__launch_bounds__(DIM_X* DIM_Y) __global__
  void csrsm_transpose(T m,
                       T n,
                       const U* __restrict__ A,
                       T lda,
                       U* __restrict__ B,
                       T ldb)
{
  T lid = threadIdx.x & (DIM_X - 1);
  T wid = threadIdx.x / DIM_X;

  T row_A = blockIdx.x * DIM_X + lid;
  T row_B = blockIdx.x * DIM_X + wid;

  __shared__ U sdata[DIM_X][DIM_X];

  for (int j = 0; j < n; j += DIM_X) {
    __syncthreads();

    int col_A = j + wid;

    for (int k = 0; k < DIM_X; k += DIM_Y) {
      if (row_A < m && col_A + k < n) {
        sdata[wid + k][lid] = A[row_A + lda * (col_A + k)];
      }
    }

    __syncthreads();

    int col_B = j + lid;

    for (int k = 0; k < DIM_X; k += DIM_Y) {
      if (col_B < n && row_B + k < m) {
        B[col_B + ldb * (row_B + k)] = sdata[lid][wid + k];
      }
    }
  }
}

template<unsigned int DIM_X, unsigned int DIM_Y, typename T, typename U>
__launch_bounds__(DIM_X* DIM_Y) __global__
  void csrsm_transpose_back(T m,
                            T n,
                            const U* __restrict__ A,
                            T lda,
                            U* __restrict__ B,
                            T ldb)
{
  T lid = threadIdx.x & (DIM_X - 1);
  T wid = threadIdx.x / DIM_X;

  T row_A = blockIdx.x * DIM_X + wid;
  T row_B = blockIdx.x * DIM_X + lid;

  __shared__ U sdata[DIM_X][DIM_X];

  for (int j = 0; j < n; j += DIM_X) {
    __syncthreads();

    int col_A = j + lid;

    for (int k = 0; k < DIM_X; k += DIM_Y) {
      if (col_A < n && row_A + k < m) {
        sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
      }
    }

    __syncthreads();

    int col_B = j + wid;

    for (int k = 0; k < DIM_X; k += DIM_Y) {
      if (row_B < m && col_B + k < n) {
        B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
      }
    }
  }
}
