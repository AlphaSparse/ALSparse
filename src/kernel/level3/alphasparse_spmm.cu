#include "../../format/bell2coo.h"
#include "../../format/csc2csr.h"
#include "../../format/transpose_conj_dn_mat.h"
#include "../../format/transpose_coo.h"
#include "../../format/transpose_csr.h"
#include "../../format/transpose_dn_mat.h"
#include "alphasparse.h"
#include "alphasparse_spmm_bell.h"
#include "alphasparse_spmm_coo_col.h"
#include "alphasparse_spmm_csr_col.h"
#include "alphasparse_spmm_csr_row.h"
#include <iostream>

template<typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmm_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              alphasparseOperation_t opB,
              const void* alpha,
              alphasparseSpMatDescr_t matA,
              alphasparseDnMatDescr_t matB,
              const void* beta,
              alphasparseDnMatDescr_t matC)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {
      if (matB->order == ALPHASPARSE_ORDER_ROW &&
          matC->order == ALPHASPARSE_ORDER_ROW) {
        spmm_csr_row<T, U, V, W>(handle,
                                 (T)matC->rows,
                                 (T)matC->cols,
                                 (T)matA->cols,
                                 (T)matA->nnz,
                                 *((W*)alpha),
                                 (U*)matA->val_data,
                                 (T*)matA->row_data,
                                 (T*)matA->col_data,
                                 (U*)matB->values,
                                 (T)matB->ld,
                                 *((W*)beta),
                                 (V*)matC->values,
                                 (T)matC->ld);
        return ALPHA_SPARSE_STATUS_SUCCESS;
      }
      if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        transpose_csr<T, U>(matA);
      }
      if (matB->order == ALPHASPARSE_ORDER_ROW) {
        U* dmatB_value = NULL;
        cudaMalloc((void**)&dmatB_value, sizeof(U) * matB->rows * matB->ld);
        transpose_dn_mat<T, U>(
          handle, matB->rows, matB->ld, (U*)matB->values, dmatB_value);
        cudaMemcpy(matB->values,
                   dmatB_value,
                   matB->rows * matB->ld * sizeof(U),
                   cudaMemcpyDeviceToDevice);
        matB->order = ALPHASPARSE_ORDER_COL;
        // int64_t temp = matB->cols;
        // matB->cols = matB->rows;
        // matB->rows = temp;
        matB->ld = matB->rows;
      }
      if (matC->order == ALPHASPARSE_ORDER_ROW) {
        V* dmatC_value = NULL;
        cudaMalloc((void**)&dmatC_value, sizeof(V) * matC->rows * matC->ld);
        transpose_dn_mat<T, V>(
          handle, matC->rows, matC->ld, (V*)matC->values, dmatC_value);
        cudaMemcpy(matC->values,
                   dmatC_value,
                   matC->rows * matC->ld * sizeof(V),
                   cudaMemcpyDeviceToDevice);
        // int64_t temp = matC->cols;
        // matC->cols = matC->rows;
        // matC->rows = temp;
        matC->ld = matC->rows;
      }
      T m = (T)matA->rows;
      T n = (T)matB->cols;
      T k = (T)matA->cols;
      spmm_csr_col<T, U, V, W>(handle,
                               m,
                               n,
                               k,
                               (T)matA->nnz,
                               *((W*)alpha),
                               (U*)matA->val_data,
                               (T*)matA->row_data,
                               (T*)matA->col_data,
                               (U*)matB->values,
                               (T)matB->ld,
                               *((W*)beta),
                               (V*)matC->values,
                               (T)matC->ld);
      if (matC->order == ALPHASPARSE_ORDER_ROW) {
        V* dmatC_value = NULL;
        cudaMalloc((void**)&dmatC_value, sizeof(V) * matC->rows * matC->cols);
        transpose_dn_mat<T, V>(
          handle, matC->cols, matC->rows, (V*)matC->values, dmatC_value);
        cudaMemcpy(matC->values,
                   dmatC_value,
                   matC->rows * matC->cols * sizeof(V),
                   cudaMemcpyDeviceToDevice);
      }
      break;
    }
    case ALPHA_SPARSE_FORMAT_COO: {
      if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        transpose_coo<T, U>(matA);
      }
      if (opA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        transpose_conj_coo<T, U>(matA);
      }
      if (matB->order == ALPHASPARSE_ORDER_ROW) {
        U* dmatB = NULL;
        cudaMalloc((void**)&dmatB, sizeof(U) * matB->rows * matB->ld);
        matB->order = ALPHASPARSE_ORDER_COL;
        int64_t temp = matB->cols;
        matB->cols = matB->rows;
        matB->rows = temp;
        transpose_dn_mat<T, U>(
          handle, matB->rows, matB->cols, (U*)matB->values, dmatB);
        cudaMemcpy(matB->values,
                   dmatB,
                   matB->rows * matB->cols * sizeof(U),
                   cudaMemcpyDeviceToDevice);
      }
      if (opB == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        U* dmatB = NULL;
        cudaMalloc((void**)&dmatB, sizeof(U) * matB->rows * matB->ld);
        matB->order = ALPHASPARSE_ORDER_COL;
        int64_t temp = matB->cols;
        matB->cols = matB->rows;
        matB->rows = temp;
        transpose_dn_mat<T, U>(
          handle, matB->rows, matB->cols, (U*)matB->values, dmatB);
        cudaMemcpy(matB->values,
                   dmatB,
                   matB->rows * matB->cols * sizeof(U),
                   cudaMemcpyDeviceToDevice);
      }
      if (opB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        U* dmatB = NULL;
        cudaMalloc((void**)&dmatB, sizeof(U) * matB->rows * matB->ld);
        matB->order = ALPHASPARSE_ORDER_COL;
        int64_t temp = matB->cols;
        matB->cols = matB->rows;
        matB->rows = temp;
        transpose_conj_dn_mat<T, U>(
          handle, matB->rows, matB->cols, (U*)matB->values, dmatB);
        cudaMemcpy(matB->values,
                   dmatB,
                   matB->rows * matB->cols * sizeof(U),
                   cudaMemcpyDeviceToDevice);
      }
      if (matC->order == ALPHASPARSE_ORDER_ROW) {
        V* dmatC = NULL;
        cudaMalloc((void**)&dmatC, sizeof(V) * matC->rows * matC->ld);
        int64_t temp = matC->cols;
        matC->cols = matC->rows;
        matC->rows = temp;
        transpose_dn_mat<T, V>(
          handle, matC->rows, matC->cols, (V*)matC->values, dmatC);
        cudaMemcpy(matC->values,
                   dmatC,
                   matC->rows * matC->cols * sizeof(V),
                   cudaMemcpyDeviceToDevice);
      }
      spmm_coo_col<T, U, V, W>(handle,
                               (T)matC->rows,
                               (T)matC->cols,
                               (T)matA->cols,
                               (T)matA->nnz,
                               *((W*)alpha),
                               (U*)matA->val_data,
                               (T*)matA->row_data,
                               (T*)matA->col_data,
                               (U*)matB->values,
                               (T)matB->ld,
                               *((W*)beta),
                               (V*)matC->values,
                               (T)matC->ld);
      if (matC->order == ALPHASPARSE_ORDER_ROW) {
        V* dmatC = NULL;
        cudaMalloc((void**)&dmatC, sizeof(V) * matC->rows * matC->cols);
        transpose_dn_mat<T, V>(
          handle, matC->rows, matC->cols, (V*)matC->values, dmatC);
        cudaMemcpy(matC->values,
                   dmatC,
                   matC->rows * matC->cols * sizeof(V),
                   cudaMemcpyDeviceToDevice);
      }
      break;
    }
    case ALPHA_SPARSE_FORMAT_CSC: {
      alphasparseSpMatDescr_t csr;
      alphasparseCsc2csr<T, U>(matA, csr);
      if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        transpose_csr<T, U>(csr);
      }
      if (matB->order == ALPHASPARSE_ORDER_ROW) {
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
      if (matC->order == ALPHASPARSE_ORDER_ROW) {
        V* dmatC_value = NULL;
        cudaMalloc((void**)&dmatC_value, sizeof(V) * matC->rows * matC->ld);
        transpose_dn_mat<T, V>(
          handle, matC->rows, matC->ld, (V*)matC->values, dmatC_value);
        cudaMemcpy(matC->values,
                   dmatC_value,
                   matC->rows * matC->ld * sizeof(V),
                   cudaMemcpyDeviceToDevice);
        int64_t temp = matC->cols;
        matC->cols = matC->rows;
        matC->rows = temp;
      }
      spmm_csr_col<T, U, V, W>(handle,
                               (T)matC->rows,
                               (T)matC->cols,
                               (T)csr->cols,
                               (T)csr->nnz,
                               *((W*)alpha),
                               (U*)csr->val_data,
                               (T*)csr->row_data,
                               (T*)csr->col_data,
                               (U*)matB->values,
                               (T)matB->cols,
                               *((W*)beta),
                               (V*)matC->values,
                               (T)matC->cols);
      if (matC->order == ALPHASPARSE_ORDER_ROW) {
        V* dmatC_value = NULL;
        cudaMalloc((void**)&dmatC_value, sizeof(V) * matC->rows * matC->cols);
        transpose_dn_mat<T, V>(
          handle, matC->rows, matC->cols, (V*)matC->values, dmatC_value);
        cudaMemcpy(matC->values,
                   dmatC_value,
                   matC->rows * matC->cols * sizeof(V),
                   cudaMemcpyDeviceToDevice);
      }
      break;
    }
    case ALPHA_SPARSE_FORMAT_BLOCKED_ELL: {
      alphasparseSpMatDescr_t coo;
      alphasparseBell2coo<T, U>(matA, coo);
      spmm_template<T, U, V, W>(handle, opA, opB, alpha, coo, matB, beta, matC);
      break;
    }
      return ALPHA_SPARSE_STATUS_SUCCESS;
  }
}

alphasparseStatus_t
alphasparseSpMM(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                alphasparseOperation_t opB,
                const void* alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseDnMatDescr_t matB,
                const void* beta,
                alphasparseDnMatDescr_t matC,
                alphasparseDataType computeType,
                alphasparseSpMMAlg_t alg,
                void* externalBuffer)
{
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F) {
    return spmm_template<int32_t, float, float, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
    return spmm_template<int32_t, double, double, double>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    return spmm_template<int32_t,
                         cuFloatComplex,
                         cuFloatComplex,
                         cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    return spmm_template<int32_t,
                         cuDoubleComplex,
                         cuDoubleComplex,
                         cuDoubleComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && matC->data_type == ALPHA_R_32I) {
    return spmm_template<int32_t, int8_t, int32_t, int32_t>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && matC->data_type == ALPHA_R_32F) {
    return spmm_template<int32_t, int8_t, float, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_32F) {
    return spmm_template<int32_t, half, float, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
      computeType == ALPHA_R_16F) {
    return spmm_template<int32_t, half, half, half>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
      computeType == ALPHA_R_32F) {
    return spmm_template<int32_t, half, half, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F) {
    return spmm_template<int32_t, half2, half2, cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
#if (CUDA_ARCH >= 80)
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF &&
      computeType == ALPHA_R_32F) {
    return spmm_template<int32_t, nv_bfloat16, nv_bfloat16, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
#endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpMM_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           alphasparseOperation_t opB,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnMatDescr_t matB,
                           const void* beta,
                           alphasparseDnMatDescr_t matC,
                           alphasparseDataType computeType,
                           alphasparseSpMMAlg_t alg,
                           size_t* bufferSize)
{
  *bufferSize = 4;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
