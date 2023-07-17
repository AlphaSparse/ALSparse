#pragma once

#include "alphasparse.h"

template <typename T, typename U, typename V, typename W, T UNROLL>
__global__ static void
spmv_csr_kernel(T m,
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
  T ix = blockIdx.x * blockDim.x + threadIdx.x;
  T stride = blockDim.x * gridDim.x;

  for (T i = ix; i < m; i += stride)
  {
    y[i] *= beta;
    V tmp = V{};

    if (UNROLL == 2)
    {
      V t1 = V{}, t2 = V{};
      T j = csr_row_ptr[i];
      for (; j < csr_row_ptr[i + 1] - 1; j += 2)
      {
        t1 += csr_val[j] * x[csr_col_ind[j]];
        t2 += csr_val[j + 1] * x[csr_col_ind[j + 1]];
      }
      for (; j < csr_row_ptr[i + 1]; ++j)
      {
        tmp += csr_val[j] * x[csr_col_ind[j]];
      }
      tmp += t1;
      tmp += t2;
    }
    else if (UNROLL == 4)
    {
      V t1 = V{}, t2 = V{}, t3 = V{}, t4 = V{};
      T j = csr_row_ptr[i];
      for (; j < csr_row_ptr[i + 1] - 3; j += 4)
      {
        t1 += csr_val[j] * x[csr_col_ind[j]];
        t2 += csr_val[j + 1] * x[csr_col_ind[j + 1]];
        t3 += csr_val[j + 2] * x[csr_col_ind[j + 2]];
        t4 += csr_val[j + 3] * x[csr_col_ind[j + 3]];
      }
      for (; j < csr_row_ptr[i + 1]; ++j)
      {
        tmp += csr_val[j] * x[csr_col_ind[j]];
      }
      tmp += t1;
      tmp += t2;
      tmp += t3;
      tmp += t4;
    }
    else if (UNROLL == 8)
    {
      V t1 = V{}, t2 = V{}, t3 = V{}, t4 = V{}, t5 = V{}, t6 = V{}, t7 = V{}, t8 = V{};
      T j = csr_row_ptr[i];
      for (; j < csr_row_ptr[i + 1] - 7; j += 8)
      {
        t1 += csr_val[j] * x[csr_col_ind[j]];
        t2 += csr_val[j + 1] * x[csr_col_ind[j + 1]];
        t3 += csr_val[j + 2] * x[csr_col_ind[j + 2]];
        t4 += csr_val[j + 3] * x[csr_col_ind[j + 3]];
        t5 += csr_val[j + 4] * x[csr_col_ind[j + 4]];
        t6 += csr_val[j + 5] * x[csr_col_ind[j + 5]];
        t7 += csr_val[j + 6] * x[csr_col_ind[j + 6]];
        t8 += csr_val[j + 7] * x[csr_col_ind[j + 7]];
      }
      for (; j < csr_row_ptr[i + 1]; ++j)
      {
        tmp += csr_val[j] * x[csr_col_ind[j]];
      }
      tmp += t1;
      tmp += t2;
      tmp += t3;
      tmp += t4;
      tmp += t5;
      tmp += t6;
      tmp += t7;
      tmp += t8;
    }
    else
    {
#pragma unroll UNROLL
      for (T j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
      {
        tmp += csr_val[j] * x[csr_col_ind[j]];
      }
    }
    y[i] += alpha * tmp;
  }
}

// template <typename T, typename U, typename V, typename W>
// __global__ static void
// spmv_csr_kernel(T m,
//                 T n,
//                 T nnz,
//                 const W alpha,
//                 const U *csr_val,
//                 const T *csr_row_ptr,
//                 const T *csr_col_ind,
//                 const U *x,
//                 const W beta,
//                 V *y)
// {
//   int ix = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (T i = ix; i < m; i += stride)
//   {
//     y[i] *= beta;
//     V tmp = {};
//     for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
//     {
//       tmp += csr_val[j] * x[csr_col_ind[j]];
//     }
//     y[i] += alpha * tmp;
//   }
// }

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmv_csr_scalar(alphasparseHandle_t handle,
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
  const int threadPerBlock = 1024;
  const int blockPerGrid = (m - 1) / threadPerBlock + 1;
  spmv_csr_kernel<T, U, V, W, 1><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
      m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}