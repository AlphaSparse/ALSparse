#include "alphasparse.h"
#include <thrust/scan.h>

template<typename T>
__global__ static void
gemm_nnz_per_row(T m,
                 T n,
                 const T* csr_row_ptr_A,
                 const T* csr_col_ind_A,
                 const T* csr_row_ptr_B,
                 const T* csr_col_ind_B,
                 const T* csr_row_ptr_D,
                 const T* csr_col_ind_D,
                 T* row_nnz)
{
  T tid = blockIdx.x * blockDim.x + threadIdx.x;
  T stride = blockDim.x * gridDim.x;

  const T trunk_size = 2048;
  bool flag[trunk_size];
  T trunk = 0;
  // printf("m:%d, blockDim.x:%d, gridDim.x:%d\n", m, blockDim.x, gridDim.x);
  for (T ar = tid; ar < m; ar += stride) {
    while (trunk < n) {
      for (T i = 0; i < trunk_size; i++)
        flag[i] = false;

      // for (T di = csr_row_ptr_D[ar]; di < csr_row_ptr_D[ar + 1]; di++) {
      //   T dc = csr_col_ind_D[di];
      //   if (dc >= trunk && dc < trunk + trunk_size) {
      //     flag[dc - trunk] = true;
      //     atomicAdd(&row_nnz[ar + 1], 1);
      //   }
      // }

      for (T ai = csr_row_ptr_A[ar]; ai < csr_row_ptr_A[ar + 1]; ai++) {
        T br = csr_col_ind_A[ai];
        for (T bi = csr_row_ptr_B[br]; bi < csr_row_ptr_B[br + 1]; bi++) {
          T bc = csr_col_ind_B[bi];
          if (bc >= trunk && bc < trunk + trunk_size && !flag[bc - trunk]) {
            atomicAdd(&row_nnz[ar + 1], 1);
            flag[bc - trunk] = true;
          }
        }
      }
      trunk += trunk_size;
    }
  }
}

template<typename T>
alphasparseStatus_t
spgemm_nnz_csr(alphasparseHandle_t handle,
               T m,
               T n,
               T k,
               T nnz_A,
               const T* csr_row_ptr_A,
               const T* csr_col_ind_A,
               T nnz_B,
               const T* csr_row_ptr_B,
               const T* csr_col_ind_B,
               T nnz_D,
               const T* csr_row_ptr_D,
               const T* csr_col_ind_D,
               T* csr_row_ptr_C,
               T* nnz_C)
{
  const int threadPerBlock = 256;
  const int blockPerGrid = (m - 1) / threadPerBlock + 1;

  cudaMemset(csr_row_ptr_C, '\0', (m + 1) * sizeof(T));
  gemm_nnz_per_row<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
    m,
    n,
    csr_row_ptr_A,
    csr_col_ind_A,
    csr_row_ptr_B,
    csr_col_ind_B,
    csr_row_ptr_D,
    csr_col_ind_D,
    csr_row_ptr_C);
  
  thrust::inclusive_scan(thrust::device, csr_row_ptr_C, csr_row_ptr_C + m + 1, csr_row_ptr_C); // in-place scan
  cudaMemcpyAsync(nnz_C, csr_row_ptr_C + m, sizeof(T), cudaMemcpyDeviceToHost);
  // std::cout << "C_nnz2: " << *nnz_C << std::endl;
  // T* row_ptr = (T*)alpha_malloc((m + 1) * sizeof(T));
  // cudaMemcpy(
  //   row_ptr, csr_row_ptr_C, (m + 1) * sizeof(T), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < (m + 1); i++) {
  //   printf("+%d+, ", row_ptr[i]);
  // }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
