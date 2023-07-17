#include "alphasparse.h"
#include "alphasparse_bsrmm.h"
#include <iostream>

template<typename T>
alphasparseStatus_t
bsrmm_template(alphasparseHandle_t handle,
              alphasparseDirection_t dirA,
              alphasparseOperation_t transB,
              int mb,
              int n,
              int kb,
              int nnzb,
              const void * alpha,
              const alphasparseMatDescr_t descrA,
              const void* bsrValA,
              const int* bsrRowPtrA,
              const int* bsrColIndA,
              int blockDim,
              const void* B,
              int ldb,
              const void* beta,
              void* C,
              int ldc)
{
  if(transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    bsrmm_plain<<<dim3(64), dim3(256), 0, handle->stream>>>(dirA, mb, n, kb, nnzb, *((T*)alpha), descrA, (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, (T*)B, ldb, *((T*)beta), (T*)C, ldc);
  else
    bsrmm_trans_plain<<<dim3(64), dim3(256), 0, handle->stream>>>(dirA, mb, n, kb, nnzb, *((T*)alpha), descrA, (T*)bsrValA, bsrRowPtrA, bsrColIndA, blockDim, (T*)B, ldb, *((T*)beta), (T*)C, ldc);
  
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const float* alpha,
                  const alphasparseMatDescr_t descrA,
                  const float* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const float* B,
                  int ldb,
                  const float* beta,
                  float* C,
                  int ldc)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsrmm_template<float>(handle, dirA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const double* alpha,
                  const alphasparseMatDescr_t descrA,
                  const double* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const double* B,
                  int ldb,
                  const double* beta,
                  double* C,
                  int ldc)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsrmm_template<double>(handle, dirA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const void* alpha,
                  const alphasparseMatDescr_t descrA,
                  const void* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const void* B,
                  int ldb,
                  const void* beta,
                  void* C,
                  int ldc)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsrmm_template<cuFloatComplex>(handle, dirA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const void* alpha,
                  const alphasparseMatDescr_t descrA,
                  const void* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const void* B,
                  int ldb,
                  const void* beta,
                  void* C,
                  int ldc)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // const int blocks_per_row = nnzb / mb; 
  bsrmm_template<cuDoubleComplex>(handle, dirA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}