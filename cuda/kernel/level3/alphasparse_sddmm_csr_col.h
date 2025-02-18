#include "alphasparse.h"
#include "alphasparse/types.h" 

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

template <typename T, typename U, int BLOCK>
__global__ static void sddmm_device(T m, T n, T k, U *a, const T lda, U *b, const T ldb,
                      U alpha, U beta,
                      const T* __restrict__ csr_row_ptr,
                      const T* __restrict__ csr_col_ind,
                      U* __restrict__ csr_val) {

  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  T num_w = stride / BLOCK;
  T w_id  = tid / BLOCK;
  T w_tid = tid % BLOCK;
  long lda_ = lda;
  long ldb_ = ldb;
  for(long row = w_id; row < m; row += num_w)
  {
    for(T r = csr_row_ptr[row] + w_tid; r < csr_row_ptr[row + 1]; r+= BLOCK)
    {
      long col = csr_col_ind[r];
      U sum = U{};
      for (long i = 0; i < k; i++)
      {
        if(i * lda_ + row >= long(m) * long(k)) printf("w_id %d w_tid %d i %d lda %d row %d m %d k %d\n", w_id, w_tid, i, lda, row, m, k);
        assert(i * lda_ + row < long(m) * long(k));
        sum += a[i * lda_ + row] * b[col * ldb_ + i];
      }

      csr_val[r] = alpha * sum + beta * csr_val[r]; 
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
  // gemm_device<T, U><<<4, 256, 0, handle->stream>>>
  //                       (m,
  //                        n,
  //                        k,
  //                        matA,
  //                        lda,
  //                        matB,
  //                        ldb,
  //                        buffer,
  //                        ld_buffer);

  // const T BLOCKSIZE = 256;
  // const T GRIDSIZE = 32;
  // csr_hadamard_prod_plain_device<T, U, W>
  //   <<<dim3(GRIDSIZE),
  //      dim3(BLOCKSIZE),
  //      0,
  //      handle->stream>>>(m,
  //                        n,
  //                        k,
  //                        nnz,
  //                        alpha,                         
  //                        buffer,
  //                        ld_buffer,
  //                        beta,
  //                        csr_row_ptr,
  //                        csr_col_ind,
  //                        csr_val);
  printf("m %d n %d k %d lda %d ldb %d\n", m, n, k, lda, ldb);
  sddmm_device<T, U, 8><<<160, 256, 0, handle->stream>>>
                        (m,
                         n,
                         k,
                         matA,
                         lda,
                         matB,
                         ldb,
                         alpha,
                         beta,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
