#include "../../format/transpose_conj_dn_mat.h"
#include "../../format/transpose_csr.h"
#include "../../format/transpose_dn_mat.h"
#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_sddmm_csr_col.h"
#include <iostream>
#include "alphasparse/common.h"

template<typename T, typename U, typename W>
alphasparseStatus_t
sddmm_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              alphasparseOperation_t opB,
              const void* alpha,
              alphasparseDnMatDescr_t matA,
              alphasparseDnMatDescr_t matB,
              const void* beta,
              alphasparseSpMatDescr_t matC,
              alphasparseDataType computeType,
              alphasparseSDDMMAlg_t alg,
              void* externalBuffer)
{
  T ld_buffer = matA->rows;
  if(matC->format != ALPHA_SPARSE_FORMAT_CSR)
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  if (opA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE || opB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    
  if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE || matA->order == ALPHASPARSE_ORDER_ROW) {
    U* dmatA_value = NULL;
    cudaMalloc((void**)&dmatA_value, sizeof(U) * matA->rows * matA->ld);
    transpose_dn_mat<T, U>(
      handle, matA->rows, matA->ld, (U*)matA->values, dmatA_value);
    cudaMemcpy(matA->values,
                dmatA_value,
                matA->rows * matA->ld * sizeof(U),
                cudaMemcpyDeviceToDevice);
    matA->order = ALPHASPARSE_ORDER_COL;
    int64_t temp = matA->cols;
    matA->cols = matA->rows;
    matA->rows = temp;
  }
  
  if (opB == ALPHA_SPARSE_OPERATION_TRANSPOSE ||matB->order == ALPHASPARSE_ORDER_ROW) {
    U* dmatB_value = NULL;
    cudaMalloc((void**)&dmatB_value, sizeof(U) * matB->rows * matB->ld);
    transpose_dn_mat<T, U>(
      handle, matB->rows, matB->ld, (U*)matB->values, dmatB_value);
    cudaMemcpy(matB->values,
                dmatB_value,
                matB->rows * matB->ld * sizeof(U),
                cudaMemcpyDeviceToDevice);
    matB->order = ALPHASPARSE_ORDER_COL;
    int64_t temp = matB->cols;
    matB->cols = matB->rows;
    matB->rows = temp;
  }
  
  sddmm_csr_col<T, U, W>(handle,
                        (T)matC->rows,
                        (T)matC->cols,
                        (T)matA->cols,
                        (T)matC->nnz,
                        *((W*)alpha),
                        (U*)matA->values,
                        (T)matA->ld,
                        (U*)matB->values,
                        (T)matB->ld,
                        *((W*)beta),                              
                        (U*)matC->val_data,
                        (T*)matC->row_data,
                        (T*)matC->col_data,
                        (U*)externalBuffer,
                        (T)ld_buffer);
      
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSDDMM(alphasparseHandle_t handle,
                  alphasparseOperation_t opA,
                  alphasparseOperation_t opB,
                  const void* alpha,
                  alphasparseDnMatDescr_t matA,
                  alphasparseDnMatDescr_t matB,
                  const void* beta,
                  alphasparseSpMatDescr_t matC,
                  alphasparseDataType computeType,
                  alphasparseSDDMMAlg_t alg,
                  void* externalBuffer)
{
  // single real ; i32
  if (matC->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F) {
    return sddmm_template<int32_t, float, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
  }
  if (matC->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
    return sddmm_template<int32_t, double, double>(
      handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
  }
  if (matC->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    return sddmm_template<int32_t,
                         cuFloatComplex,
                         cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
  }
  if (matC->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    return sddmm_template<int32_t,
                         cuDoubleComplex,
                         cuDoubleComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
  }
  
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSDDMM_preprocess(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           alphasparseOperation_t opB,
                           const void* alpha,
                           alphasparseDnMatDescr_t matA,
                           alphasparseDnMatDescr_t matB,
                           const void* beta,
                           alphasparseSpMatDescr_t matC,
                           alphasparseDataType computeType,
                           alphasparseSDDMMAlg_t alg,
                           void* externalBuffer)
{
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSDDMM_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           alphasparseOperation_t opB,
                           const void* alpha,
                           alphasparseDnMatDescr_t matA,
                           alphasparseDnMatDescr_t matB,
                           const void* beta,
                           alphasparseSpMatDescr_t matC,
                           alphasparseDataType computeType,
                           alphasparseSDDMMAlg_t alg,
                           size_t* bufferSize)
{
  long long t = matA->rows * matB->cols;
  t = 1;
  if(computeType == ALPHA_R_32F) *bufferSize = t * sizeof(float);
  else if(computeType == ALPHA_R_64F) *bufferSize = t * sizeof(double);
  else if(computeType == ALPHA_C_32F) *bufferSize = t * sizeof(float) * 2;
  else if(computeType == ALPHA_C_64F) *bufferSize = t * sizeof(double) * 2;
  else return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
