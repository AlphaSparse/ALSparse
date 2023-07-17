#include "alphasparse.h"

template <typename T, typename U>
__global__ static void gemm_device(const T m, const T n, const T k, U *a, const T lda, U *b, const T ldb,
                      U *c, const T ldc) {

  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < m; i+=stride)
  {        
    for (int j = 0; j < n; j++)
    {
      for(int p = 0; p < k; p++)
      {
        long long inda = i + p * lda;
        long long indb = p + j * ldb;
        long long indc = i + j * ldc;
        c[indc] += a[inda] * b[indb];
      }
    }
  }
}

template <typename T, typename U>
__global__ static void print_device(const T size, U *a) {

  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < size; i+= stride)
    printf("array %d : %f \n", i, a[i]);
}

template<typename T,
         typename U,
         typename W>
static __global__ void
  csr_hadamard_prod_plain_device(T M,
                                T N,
                                T K,
                                T nnz,
                                W alpha,
                                U* __restrict__ mat,
                                T ldc,
                                W beta,
                                const T* __restrict__ csr_row_ptr,
                                const T* __restrict__ csr_col_ind,
                                U* __restrict__ csr_val)                            
{
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int rows = tid; rows < M; rows += stride)
  {
    for(int r = csr_row_ptr[rows]; r < csr_row_ptr[rows + 1]; r ++)
    {
      int col = csr_col_ind[r];
      csr_val[r] = alpha * mat[rows + col * ldc] + beta * csr_val[r];
    }
  }
}

template<typename T, typename U, typename W>
static alphasparseStatus_t
sddmm_csr_col(alphasparseHandle_t handle,
             T m,
             T n,
             T k,
             T nnz,
             W alpha,
             U* matA,
             const T lda,             
             U* matB,
             const T ldb,
             const W beta,
             U* csr_val,
             const T* csr_row_ptr,
             const T* csr_col_ind,
             U* buffer,
             const T ld_buffer)
{  
  gemm_device<T, U><<<4, 256, 0, handle->stream>>>
                        (m,
                         n,
                         k,
                         matA,
                         lda,
                         matB,
                         ldb,
                         buffer,
                         ld_buffer);

  const T BLOCKSIZE = 256;
  const T GRIDSIZE = 32;
  csr_hadamard_prod_plain_device<T, U, W>
    <<<dim3(GRIDSIZE),
       dim3(BLOCKSIZE),
       0,
       handle->stream>>>(m,
                         n,
                         k,
                         nnz,
                         alpha,                         
                         buffer,
                         ld_buffer,
                         beta,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
