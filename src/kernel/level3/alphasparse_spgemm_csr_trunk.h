#include "alphasparse.h"
#include <iostream>

template<typename T, typename U>
__global__ static void
spgemm_blk(T m,
           T n,
           T k,
           const U alpha,
           const U* csr_val_A,
           const T* csr_row_ptr_A,
           const T* csr_col_ind_A,
           const U* csr_val_B,
           const T* csr_row_ptr_B,
           const T* csr_col_ind_B,
           const U beta,
           const U* csr_val_D,
           const T* csr_row_ptr_D,
           const T* csr_col_ind_D,
           U* csr_val_C,
           const T* csr_row_ptr_C,
           T* csr_col_ind_C,
           const T trunk_size)
{
  T tid = threadIdx.x;
  T stride = blockDim.x;

  extern __shared__ char shr[];
  U* values = reinterpret_cast<U*>(shr);
  T* write_back = reinterpret_cast<T*>(values + trunk_size);
  T trunk = 0;
  // for (T ar = tid; ar < m; ar += stride)
  T ar = blockIdx.x;
  while (trunk < n) {
    for (T i = tid; i < trunk_size; i += stride) {
      values[i] = U{};
      write_back[i] = 0;
    }
    __syncthreads();

    // for (T di = csr_row_ptr_D[ar] + tid; di < csr_row_ptr_D[ar + 1];
    //      di += stride) {
    //   T dc = csr_col_ind_D[di];
    //   values[dc] = beta * csr_val_D[di];
    //   write_back[dc] = 1;
    // }
    // __syncthreads();

    for (T ai = csr_row_ptr_A[ar] + tid; ai < csr_row_ptr_A[ar + 1];
         ai += stride) {
      T br = csr_col_ind_A[ai];
      U av = csr_val_A[ai];
      U tmp = alpha * csr_val_A[ai];

      for (T bi = csr_row_ptr_B[br]; bi < csr_row_ptr_B[br + 1]; bi++) {
        T bc = csr_col_ind_B[bi];
        U bv = csr_val_B[bi];
        if (bc >= trunk && bc < trunk + trunk_size) {
          // alpha_madde(values[bc], tmp, bv);
          U t = tmp * bv;
          atomicAdd(&values[bc - trunk], t);
          write_back[bc - trunk] = 1;
        }
      }
    }
    __syncthreads();

    // in-place prefix sum
    T n64 = 1;
    T stop = 2;
    T t_stop = 1;
    T i;

    while (n64 < trunk_size)
      n64 = n64 << 1;
    n64 = n64 >> 1;

    if (n64 != 0) {
      while (stop <= n64) {
        for (i = tid; i < n64; i += stride) {
          if (i % stop >= t_stop) {
            // atomicAdd(&write_back[i], write_back[i - i % t_stop - 1]);
            write_back[i] += write_back[i - i % t_stop - 1];
          }
        }
        __syncthreads();

        stop = stop << 1;
        t_stop = t_stop << 1;
      }
    } else
      n64++;

    if (tid == 0) {
      for (T i = n64; i < trunk_size; i++) {
        // atomicAdd(&write_back[i], write_back[i - 1]);
        write_back[i] = write_back[i] + write_back[i - 1];
      }
    }

    __syncthreads();

    T index = csr_row_ptr_C[ar];
    for (T c = tid; c < trunk_size; c += stride) {
      if (c + trunk == 0 && write_back[c]) {
        csr_col_ind_C[index] = c;
        csr_val_C[index] = values[c];
        // if(index == 116)
        // printf("\n*/*/*/%d,%d*/*/*/*/*\n",c,ar);
      } else {
        if (write_back[c] - write_back[c - 1] && values[c] != U{}) {
          csr_col_ind_C[index + write_back[c] - 1] = c + trunk;
          csr_val_C[index + write_back[c] - 1] = values[c];
          //   if(index + write_back[c] - 1 == 116)
          // printf("\n-*-*-*%d,%d,%f-*-*-*\n",c,ar,values[c]);
          // printf("-%f-",values[c]);
        }
      }
    }

    // TODO scan + write
    //  if (tid == 0)
    //  {
    //      T index = csr_row_ptr_C[ar];
    //      for (T c = 0; c < n; c++)
    //      {
    //          if (write_back[c])
    //          {
    //              csr_col_ind_C[index] = c;
    //              csr_val_C[index] = values[c];
    //              index += 1;
    //          }
    //      }
    //  }
    trunk += trunk_size;
  }
}

template<typename T, typename U>
alphasparseStatus_t
spgemm_csr(alphasparseHandle_t handle,
           T m,
           T n,
           T k,
           const U alpha,
           T nnz_A,
           const U* csr_val_A,
           const T* csr_row_ptr_A,
           const T* csr_col_ind_A,
           T nnz_B,
           const U* csr_val_B,
           const T* csr_row_ptr_B,
           const T* csr_col_ind_B,
           U beta,
           T nnz_D,
           U* csr_val_D,
           T* csr_row_ptr_D,
           T* csr_col_ind_D,
           U* csr_val_C,
           T* csr_row_ptr_C,
           T* csr_col_ind_C)
{

  const T threadPerBlock = 256;
  const T blockPerGrid = (m - 1) / threadPerBlock + 1;
  const T trunk_size = 512;
  spgemm_blk<<<m,
               threadPerBlock,
               trunk_size * (sizeof(U) + sizeof(T)),
               handle->stream>>>(m,
                                 n,
                                 k,
                                 alpha,
                                 csr_val_A,
                                 csr_row_ptr_A,
                                 csr_col_ind_A,
                                 csr_val_B,
                                 csr_row_ptr_B,
                                 csr_col_ind_B,
                                 beta,
                                 csr_val_D,
                                 csr_row_ptr_D,
                                 csr_col_ind_D,
                                 csr_val_C,
                                 csr_row_ptr_C,
                                 csr_col_ind_C,
                                 trunk_size);
  // csr_val_D = dCval;
  // cudaMemcpy(csr_val_D, dCval, nnz_C * sizeof(U), cudaMemcpyDeviceToDevice);
  // cudaMemcpy(
  //   csr_row_ptr_D, dCptr, (m + 1) * sizeof(T), cudaMemcpyDeviceToDevice);
  // cudaMemcpy(csr_col_ind_D, dCval, nnz_C * sizeof(T),
  // cudaMemcpyDeviceToDevice);
  float* matC_ict;
  matC_ict = (float*)malloc(20 * sizeof(float));
  cudaMemcpy(matC_ict, csr_val_C, 20 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "\n===================" << std::endl;
  for (int i = 0; i < 20; i++) {
    printf("%f, ", matC_ict[i]);
  }
  std::cout << "\n===================\n" << std::endl;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
