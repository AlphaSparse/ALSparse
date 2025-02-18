#include "alphasparse.h"
#include "alphasparse/types.h" 

template<typename T, typename U, typename V, typename W>
__global__ static void
spmm_csr_device(T m,
                T n,
                T k,
                T nnz,
                W alpha,
                const U* csr_val,
                const T* csr_row_ptr,
                const T* csr_col_ind,
                const U* x,
                T ldx,
                W beta,
                V* y,
                T ldy)
{
  T bid = blockIdx.x;
  T b_stride = blockDim.x;
  T tid = threadIdx.x;
  T t_stride = blockDim.x;
  for (T cc = bid; cc < n; cc += b_stride) {
    for (T cr = tid; cr < m; cr += t_stride) {
      U ctmp = {};
      for (T ai = csr_row_ptr[cr]; ai < csr_row_ptr[cr + 1]; ++ai) {
        ctmp += csr_val[ai] * x[index2(cc, csr_col_ind[ai], ldx)];
      }
      y[index2(cc, cr, ldy)] *= beta;
      y[index2(cc, cr, ldy)] += alpha * ctmp;
    }
  }
}

template<typename T, typename U, typename V, typename W>
static alphasparseStatus_t
spmm_csr(alphasparseHandle_t handle,
         T m,
         T n,
         T k,
         T nnz,
         W alpha,
         const U* csr_val,
         const T* csr_row_ptr,
         const T* csr_col_ind,
         const U* matB,
         T ldb,
         W beta,
         V* matC,
         T ldc)
{
  const int threadPerBlock = 256;
  const int blockPerGrid = min(32, (threadPerBlock + k - 1) / threadPerBlock);
  spmm_csr_device<T, U, V, W>
    <<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(
      m,
      n,
      k,
      nnz,
      alpha,
      csr_val,
      csr_row_ptr,
      csr_col_ind,
      matB,
      ldb,
      beta,
      matC,
      ldc);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}