#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_gemvi.h"
#include <iostream>

template<typename T>
alphasparseStatus_t
gemvi_template(alphasparseHandle_t handle,
              alphasparseOperation_t transA,
              int m,
              int n,
              const void* alpha,
              const void* A,
              int lda,
              int nnz,
              const void* x,
              const int* xInd,
              const void* beta,
              void* y,
              alphasparseIndexBase_t idxBase,
              void* pBuffer)
{
  if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
  {
    gemvi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(m, n, *((T*)alpha), (T*)A, lda, nnz, (T*)x, xInd, *((T*)beta), (T*)y, idxBase, pBuffer);
  }
  else if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
  {
    gemvi_trans_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(m, n, *((T*)alpha), (T*)A, lda, nnz, (T*)x, xInd, *((T*)beta), (T*)y, idxBase, pBuffer);
  }
  else
  {
    gemvi_trans_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(m, n, *((T*)alpha), (T*)A, lda, nnz, (T*)x, xInd, *((T*)beta), (T*)y, idxBase, pBuffer);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSgemvi(alphasparseHandle_t handle,
              alphasparseOperation_t transA,
              int m,
              int n,
              const float* alpha,
              const float* A,
              int lda,
              int nnz,
              const float* x,
              const int* xInd,
              const float* beta,
              float* y,
              alphasparseIndexBase_t idxBase,
              void* pBuffer)
{
  gemvi_template<float>(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgemvi(alphasparseHandle_t handle,
              alphasparseOperation_t transA,
              int m,
              int n,
              const double* alpha,
              const double* A,
              int lda,
              int nnz,
              const double* x,
              const int* xInd,
              const double* beta,
              double* y,
              alphasparseIndexBase_t idxBase,
              void* pBuffer)
{
  gemvi_template<double>(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgemvi(alphasparseHandle_t handle,
              alphasparseOperation_t transA,
              int m,
              int n,
              const void* alpha,
              const void* A,
              int lda,
              int nnz,
              const void* x,
              const int* xInd,
              const void* beta,
              void* y,
              alphasparseIndexBase_t idxBase,
              void* pBuffer)
{
  gemvi_template<cuFloatComplex>(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgemvi(alphasparseHandle_t handle,
              alphasparseOperation_t transA,
              int m,
              int n,
              const void* alpha,
              const void* A,
              int lda,
              int nnz,
              const void* x,
              const int* xInd,
              const void* beta,
              void* y,
              alphasparseIndexBase_t idxBase,
              void* pBuffer)
{
  gemvi_template<cuDoubleComplex>(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
