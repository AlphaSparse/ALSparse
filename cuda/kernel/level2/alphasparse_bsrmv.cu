#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_bsrmv.h"
#include <iostream>

template<typename T>
alphasparseStatus_t
bsrmv_template(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              int mb,
              int nb,
              int nnzb,
              const void* alpha,
              const void* bsrVal,
              const int* bsrRowPtr,
              const int* bsrColInd,
              int blockDim,
              const void* x,
              const void* beta,
              void* y)
{
  bsr_gemv_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nb, nnzb, *((T*)alpha), (T*)bsrVal, bsrRowPtr, bsrColInd, blockDim, (T*)x, *((T*)beta), (T*)y);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
alphasparseStatus_t
alphasparseSbsrmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int mb,
              int nb,
              int nnzb,
              const float* alpha,
              const alphasparseMatDescr_t descr,
              const float* bsrVal,
              const int* bsrRowPtr,
              const int* bsrColInd,
              int blockDim,
              const float* x,
              const float* beta,
              float* y)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(trans != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descr->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsr_gemv_plain<float><<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nb, nnzb, *alpha, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, *beta, y);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int mb,
              int nb,
              int nnzb,
              const double* alpha,
              const alphasparseMatDescr_t descr,
              const double* bsrVal,
              const int* bsrRowPtr,
              const int* bsrColInd,
              int blockDim,
              const double* x,
              const double* beta,
              double* y)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(trans != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descr->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsr_gemv_plain<double><<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nb, nnzb, *alpha, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, *beta, y);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int mb,
              int nb,
              int nnzb,
              const void* alpha,
              const alphasparseMatDescr_t descr,
              const void* bsrVal,
              const int* bsrRowPtr,
              const int* bsrColInd,
              int blockDim,
              const void* x,
              const void* beta,
              void* y)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(trans != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descr->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsrmv_template<cuFloatComplex>(handle, dir, mb, nb, nnzb, alpha, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int mb,
              int nb,
              int nnzb,
              const void* alpha,
              const alphasparseMatDescr_t descr,
              const void* bsrVal,
              const int* bsrRowPtr,
              const int* bsrColInd,
              int blockDim,
              const void* x,
              const void* beta,
              void* y)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(trans != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descr->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsrmv_template<cuDoubleComplex>(handle, dir, mb, nb, nnzb, alpha, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}