#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_bsrsv2.h"
#include <iostream>

template<typename T>
alphasparseStatus_t
bsrsv2_template(alphasparseHandle_t handle,
              alphasparseDirection_t dir,
              alphasparseOperation_t transA,
              int mb,
              int nnzb,
              const void* alpha,
              const alphasparseMatDescr_t descrA,
              const void* bsrValA,
              const int* bsrRowPtrA,
              const int* bsrColIndA,
              int blockDim,
              alpha_bsrsv2Info_t info,
              const void* x,
              void* y,
              alphasparseSolvePolicy_t policy,
              void* pBuffer)
{
  if(policy == ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL)
  {
    if(descrA->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    {
      if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
      {
        if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
          bsrsv2_n_lo_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {
          bsrsv2_n_hi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else
          bsrsv2_n_hi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
      }
      else
      {
        if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
          bsrsv2_u_lo_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {
          bsrsv2_u_hi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else
          bsrsv2_u_hi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
      }
    }
    else
    {
      if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
      {
        if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
          bsrsv2_n_hi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {
          bsrsv2_n_lo_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else
          bsrsv2_n_lo_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
      }
      else
      {
        if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
          bsrsv2_u_hi_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {
          bsrsv2_u_lo_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
        }
        else
          bsrsv2_u_lo_plain<<<dim3(1), dim3(1), 0, handle->stream>>>(dir, mb, nnzb, *((T*)alpha), (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, (T*)x, (T*)y, (T*)pBuffer);
      }
    }    
  }
  else
  {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrsv2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        int mb,
                        int nnzb,
                        const float* alpha,
                        const alphasparseMatDescr_t descrA,
                        const float* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrsv2Info_t info,
                        const float* x,
                        float* y,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsv2_template<float>(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, x, y, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrsv2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        int mb,
                        int nnzb,
                        const double* alpha,
                        const alphasparseMatDescr_t descrA,
                        const double* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrsv2Info_t info,
                        const double* x,
                        double* y,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsv2_template<double>(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, x, y, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrsv2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        int mb,
                        int nnzb,
                        const void* alpha,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrsv2Info_t info,
                        const void* x,
                        void* y,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsv2_template<cuFloatComplex>(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, x, y, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrsv2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        int mb,
                        int nnzb,
                        const void* alpha,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrsv2Info_t info,
                        const void* x,
                        void* y,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsv2_template<cuDoubleComplex>(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, x, y, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              float* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(float);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              double* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(double);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(float) * 2;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(double) * 2;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrsv2_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              float* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrsv2_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              double* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrsv2_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrsv2_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}