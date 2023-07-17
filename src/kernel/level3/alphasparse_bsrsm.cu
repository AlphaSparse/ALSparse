#include "alphasparse.h"
#include "alphasparse_bsrsm_lo.h"
#include "alphasparse_bsrsm_up.h"
#include "alphasparse_bsrsm_conj_trans.h"
#include <iostream>

template<typename T>
alphasparseStatus_t
bsrsm2_template(alphasparseHandle_t handle,
                alphasparseDirection_t dirA,
                alphasparseOperation_t transX,
                int mb,
                int n,
                int nnzb,
                const void* alpha,
                const alphasparseMatDescr_t descrA,
                const void* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int blockDim,
                alpha_bsrsm2Info_t info,
                const void* B,
                int ldb,
                void* X,
                int ldx,
                alphasparseSolvePolicy_t policy,
                void* pBuffer)
{
  if(policy == ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL)
  {
    if(descrA->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    {
      if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
      {
        printf("dir %d\n",dirA);
        if(transX == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {          
          bsrsm2_n_lo_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
        }
        else if(transX == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {        
          bsrsm2_n_lo_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
          return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else          
          bsrsm2_n_lo_conj_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
          return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
      else
      {
        if(transX == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
          bsrsm2_u_lo_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
        }
        else if(transX == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {          
          bsrsm2_u_lo_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
          return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else          
          bsrsm2_u_lo_conj_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
          return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else//UPPER
    {
      if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
      {
        printf("dir %d\n",dirA);
        if(transX == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {          
          bsrsm2_n_up_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
        }
        else if(transX == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {        
          bsrsm2_n_up_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
        }
        else          
          bsrsm2_n_up_conj_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
      }
      else
      {
        if(transX == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
          bsrsm2_u_up_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
        }
        else if(transX == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        {          
          bsrsm2_u_up_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
        }
        else          
          bsrsm2_u_up_conj_trans_plain<<<dim3(1), dim3(16), 0, handle->stream>>>(dirA, mb, n, nnzb, *((T*)alpha), (T*)bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (T*)B, ldb, (T*)X, ldx, (T*)pBuffer);
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
alphasparseSbsrsm2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        alphasparseOperation_t transX,
                        int mb,
                        int n,
                        int nnzb,
                        const float* alpha,
                        const alphasparseMatDescr_t descrA,
                        const float* bsrSortedVal,
                        const int* bsrSortedRowPtr,
                        const int* bsrSortedColInd,
                        int blockDim,
                        alpha_bsrsm2Info_t info,
                        const float* B,
                        int ldb,
                        float* X,
                        int ldx,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transX == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsm2_template<float>(handle, dirA, transX, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, B, ldb, X, ldx, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrsm2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        alphasparseOperation_t transX,
                        int mb,
                        int n,
                        int nnzb,
                        const double* alpha,
                        const alphasparseMatDescr_t descrA,
                        const double* bsrSortedVal,
                        const int* bsrSortedRowPtr,
                        const int* bsrSortedColInd,
                        int blockDim,
                        alpha_bsrsm2Info_t info,
                        const double* B,
                        int ldb,
                        double* X,
                        int ldx,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transX == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsm2_template<double>(handle, dirA, transX, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, B, ldb, X, ldx, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrsm2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        alphasparseOperation_t transX,
                        int mb,
                        int n,
                        int nnzb,
                        const void* alpha,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrSortedVal,
                        const int* bsrSortedRowPtr,
                        const int* bsrSortedColInd,
                        int blockDim,
                        alpha_bsrsm2Info_t info,
                        const void* B,
                        int ldb,
                        void* X,
                        int ldx,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transX == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsm2_template<cuFloatComplex>(handle, dirA, transX, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, B, ldb, X, ldx, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrsm2_solve(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        alphasparseOperation_t transA,
                        alphasparseOperation_t transX,
                        int mb,
                        int n,
                        int nnzb,
                        const void* alpha,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrSortedVal,
                        const int* bsrSortedRowPtr,
                        const int* bsrSortedColInd,
                        int blockDim,
                        alpha_bsrsm2Info_t info,
                        const void* B,
                        int ldb,
                        void* X,
                        int ldx,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
  if(blockDim <= 1) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(transX == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  bsrsm2_template<cuDoubleComplex>(handle, dirA, transX, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, B, ldb, X, ldx, policy, pBuffer);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              float* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(float);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              double* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(double);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(float) * 2;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes)
{
  int pBufferSize = mb * blockDim;
  if(descrA->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT) pBufferSize *= 2;
  *pBufferSizeInBytes =  pBufferSize * sizeof(double) * 2;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            float* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            double* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}