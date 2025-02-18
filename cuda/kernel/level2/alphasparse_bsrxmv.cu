#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_bsrxmv.h"
#include <iostream>

template<typename T>
alphasparseStatus_t
bsrxmv_template(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              int sizeOfMask,
              int mb,
              int nb,
              int nnzb,
              const void* alpha,
              const void* bsrVal,
              const int* bsrMaskPtr,
              const int* bsrRowPtr,
              const int* bsrEndPtr,
              const int* bsrColInd,
              int blockDim,
              const void* x,
              const void* beta,
              void* y)
{
  bsr_xmv_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, sizeOfMask, mb, nb, nnzb, *((T*)alpha), (T*)bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, (T*)x, *((T*)beta), (T*)y);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
alphasparseStatus_t
alphasparseSbsrxmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int sizeOfMask,
              int mb,
              int nb,
              int nnzb,
              const float* alpha,
              const alphasparseMatDescr_t descr,
              const float* bsrVal,
              const int* bsrMaskPtr,
              const int* bsrRowPtr,
              const int* bsrEndPtr,
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
  bsr_xmv_plain<float><<<dim3(1), dim3(1), 0, handle->stream>>>(dir, sizeOfMask, mb, nb, nnzb, *alpha, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, *beta, y);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrxmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int sizeOfMask,
              int mb,
              int nb,
              int nnzb,
              const double* alpha,
              const alphasparseMatDescr_t descr,
              const double* bsrVal,
              const int* bsrMaskPtr,
              const int* bsrRowPtr,
              const int* bsrEndPtr,
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
  bsr_xmv_plain<double><<<dim3(1), dim3(1), 0, handle->stream>>>(dir, sizeOfMask, mb, nb, nnzb, *alpha, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, *beta, y);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrxmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int sizeOfMask,
              int mb,
              int nb,
              int nnzb,
              const void* alpha,
              const alphasparseMatDescr_t descr,
              const void* bsrVal,
              const int* bsrMaskPtr,
              const int* bsrRowPtr,
              const int* bsrEndPtr,
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
  bsrxmv_template<cuFloatComplex>(handle, dir, sizeOfMask, mb, nb, nnzb, alpha, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrxmv(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t trans,
              int sizeOfMask,
              int mb,
              int nb,
              int nnzb,
              const void* alpha,
              const alphasparseMatDescr_t descr,
              const void* bsrVal,
              const int* bsrMaskPtr,
              const int* bsrRowPtr,
              const int* bsrEndPtr,
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
  bsrxmv_template<cuDoubleComplex>(handle, dir, sizeOfMask, mb, nb, nnzb, alpha, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}