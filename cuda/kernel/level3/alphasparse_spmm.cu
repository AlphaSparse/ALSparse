#include "../../format/bell2coo.h"
#include "../../format/csc2csr.h"
#include "../../format/transpose_conj_dn_mat.h"
#include "../../format/transpose_coo.h"
#include "../../format/transpose_csr.h"
#include "../../format/transpose_csc.h"
#include "../../format/transpose_dn_mat.h"
#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_spmm_bell.h"
#include "alphasparse_spmm_coo_col.h"
#include "alphasparse_spmm_csr_col.h"
#include "alphasparse_spmm_csr_row.h"
#include "alphasparse_spmm_csr.h"
#include "alphasparse_spmm_csc.h"
#include <iostream>
#include "alphasparse/common.h"

template<typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmm_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              alphasparseOperation_t opB,
              const void* alpha,
              alphasparseSpMatDescr_t matA,
              alphasparseDnMatDescr_t matB,
              const void* beta,
              alphasparseDnMatDescr_t matC,
							alphasparseSpMMAlg_t alg,
							void *externalBuffer)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {
      if (matB->order == ALPHASPARSE_ORDER_ROW &&
          matC->order == ALPHASPARSE_ORDER_ROW) {
				switch (alg) {
					case ALPHASPARSE_SPMM_CSR_ALG1: {
						csrspmm_rb_sr<T, U, V, W, true>(handle,
																				(T)matC->rows,
																				(T)matC->cols,
																				(T)matA->cols,
																				(T)matA->nnz,
																				*((W*)alpha),
																				(T*)matA->row_data,
																				(T*)matA->col_data,
																				(U*)matA->val_data,
																				(U*)matB->values,
																				(T)matB->ld,
																				*((W*)beta),
																				(V*)matC->values,
																				(T)matC->ld,
																				externalBuffer);
						break;
					}
					default:
					case ALPHASPARSE_SPMM_CSR_ALG2: {
						csrspmm_eb_sr<T, U, V, W, true>(handle,
																				(T)matC->rows,
																				(T)matC->cols,
																				(T)matA->cols,
																				(T)matA->nnz,
																				*((W*)alpha),
																				(T*)matA->row_data,
																				(T*)matA->col_data,
																				(U*)matA->val_data,
																				(U*)matB->values,
																				(T)matB->ld,
																				*((W*)beta),
																				(V*)matC->values,
																				(T)matC->ld,
																				externalBuffer);
						break;
					}
					case ALPHASPARSE_SPMM_CSR_ALG3: {
						csrspmm_merge<T, U, V, W, true>(handle,
                                          (T)matC->rows,
                                          (T)matC->cols,
                                          (T)matA->cols,
                                          (T)matA->nnz,
                                          *((W*)alpha),
                                          (T*)matA->row_data,
                                          (T*)matA->col_data,
                                          (U*)matA->val_data,
                                          (U*)matB->values,
                                          (T)matB->ld,
                                          *((W*)beta),
                                          (V*)matC->values,
                                          (T)matC->ld,
                                          externalBuffer);
						break;
					}
					case ALPHASPARSE_SPMM_CSR_ALG4: {
						csrspmm_adaptive<T, U, V, W, true>(handle,
                                        (T)matC->rows,
                                        (T)matC->cols,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W*)alpha),
                                        (T*)matA->row_data,
                                        (T*)matA->col_data,
                                        (U*)matA->val_data,
                                        (U*)matB->values,
                                        (T)matB->ld,
                                        *((W*)beta),
                                        (V*)matC->values,
                                        (T)matC->ld,
                                        externalBuffer);
						break;
					}
					case ALPHASPARSE_SPMM_CSR_ALG5: {
						csrspmm_flat<T, U, V, W, true>(handle,
                                          (T)matC->rows,
                                          (T)matC->cols,
                                          (T)matA->cols,
                                          (T)matA->nnz,
                                          *((W*)alpha),
                                          (T*)matA->row_data,
                                          (T*)matA->col_data,
                                          (U*)matA->val_data,
                                          (U*)matB->values,
                                          (T)matB->ld,
                                          *((W*)beta),
                                          (V*)matC->values,
                                          (T)matC->ld,
                                          externalBuffer);
						break;
					}
				}
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
      switch (alg) {
        case ALPHASPARSE_SPMM_CSR_ALG1: {
          csrspmm_rb_sr<T, U, V, W, false>(handle,
                                      (T)matC->rows,
                                      (T)matC->cols,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W*)alpha),
                                      (T*)matA->row_data,
                                      (T*)matA->col_data,
                                      (U*)matA->val_data,
                                      (U*)matB->values,
                                      (T)matB->ld,
                                      *((W*)beta),
                                      (V*)matC->values,
                                      (T)matC->ld,
                                      externalBuffer);
          break;
        }
        case ALPHASPARSE_SPMM_CSR_ALG2: {
          csrspmm_eb_sr<T, U, V, W, false>(handle,
                                      (T)matC->rows,
                                      (T)matC->cols,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W*)alpha),
                                      (T*)matA->row_data,
                                      (T*)matA->col_data,
                                      (U*)matA->val_data,
                                      (U*)matB->values,
                                      (T)matB->ld,
                                      *((W*)beta),
                                      (V*)matC->values,
                                      (T)matC->ld,
                                      externalBuffer);
          break;
        }
        case ALPHASPARSE_SPMM_CSR_ALG3: {
          csrspmm_merge<T, U, V, W, false>(handle,
                                        (T)matC->rows,
                                        (T)matC->cols,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W*)alpha),
                                        (T*)matA->row_data,
                                        (T*)matA->col_data,
                                        (U*)matA->val_data,
                                        (U*)matB->values,
                                        (T)matB->ld,
                                        *((W*)beta),
                                        (V*)matC->values,
                                        (T)matC->ld,
                                        externalBuffer);
          break;
        }
        case ALPHASPARSE_SPMM_CSR_ALG4: {
          csrspmm_adaptive<T, U, V, W, false>(handle,
                                      (T)matC->rows,
                                      (T)matC->cols,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W*)alpha),
                                      (T*)matA->row_data,
                                      (T*)matA->col_data,
                                      (U*)matA->val_data,
                                      (U*)matB->values,
                                      (T)matB->ld,
                                      *((W*)beta),
                                      (V*)matC->values,
                                      (T)matC->ld,
                                      externalBuffer);
          break;
        }
        case ALPHASPARSE_SPMM_CSR_ALG5: {
          csrspmm_flat<T, U, V, W, false>(handle,
                                      (T)matC->rows,
                                      (T)matC->cols,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W*)alpha),
                                      (T*)matA->row_data,
                                      (T*)matA->col_data,
                                      (U*)matA->val_data,
                                      (U*)matB->values,
                                      (T)matB->ld,
                                      *((W*)beta),
                                      (V*)matC->values,
                                      (T)matC->ld,
                                      externalBuffer);
          break;
        }
        default: {
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
        }
      }
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
    case ALPHA_SPARSE_FORMAT_CSC: {
      if (matB->order == ALPHASPARSE_ORDER_ROW &&
          matC->order == ALPHASPARSE_ORDER_ROW) {
				switch (alg) {
					case ALPHASPARSE_SPMM_CSR_ALG1: {
            cscspmm_cb<T, U, V, W, true>(handle,
                                        (T)matC->rows,
                                        (T)matC->cols,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W*)alpha),
                                        (T*)matA->col_data,
                                        (T*)matA->row_data,
                                        (U*)matA->val_data,
                                        (U*)matB->values,
                                        (T)matB->ld,
                                        *((W*)beta),
                                        (V*)matC->values,
                                        (T)matC->ld,
                                        externalBuffer);
            break;
					}
					default:
					case ALPHASPARSE_SPMM_CSR_ALG2: {
            cscspmm_eb<T, U, V, W, true>(handle,
                                        (T)matC->rows,
                                        (T)matC->cols,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W*)alpha),
                                        (T*)matA->col_data,
                                        (T*)matA->row_data,
                                        (U*)matA->val_data,
                                        (U*)matB->values,
                                        (T)matB->ld,
                                        *((W*)beta),
                                        (V*)matC->values,
                                        (T)matC->ld,
                                        externalBuffer);
            break;
					}
					case ALPHASPARSE_SPMM_CSR_ALG4: {
            cscspmm_adaptive<T, U, V, W, true>(handle,
                                        (T)matC->rows,
                                        (T)matC->cols,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W*)alpha),
                                        (T*)matA->col_data,
                                        (T*)matA->row_data,
                                        (U*)matA->val_data,
                                        (U*)matB->values,
                                        (T)matB->ld,
                                        *((W*)beta),
                                        (V*)matC->values,
                                        (T)matC->ld,
                                        externalBuffer);
            break;
					}
				}
        return ALPHA_SPARSE_STATUS_SUCCESS;
      }
      if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        transpose_csc<T, U>(matA);
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
      switch (alg) {
        case ALPHASPARSE_SPMM_CSR_ALG1: {
          cscspmm_cb<T, U, V, W, false>(handle,
                                      (T)matC->rows,
                                      (T)matC->cols,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W*)alpha),
                                      (T*)matA->col_data,
                                      (T*)matA->row_data,
                                      (U*)matA->val_data,
                                      (U*)matB->values,
                                      (T)matB->ld,
                                      *((W*)beta),
                                      (V*)matC->values,
                                      (T)matC->ld,
                                      externalBuffer);
          break;
        }
        case ALPHASPARSE_SPMM_CSR_ALG2: {
          cscspmm_eb<T, U, V, W, false>(handle,
                                      (T)matC->rows,
                                      (T)matC->cols,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W*)alpha),
                                      (T*)matA->col_data,
                                      (T*)matA->row_data,
                                      (U*)matA->val_data,
                                      (U*)matB->values,
                                      (T)matB->ld,
                                      *((W*)beta),
                                      (V*)matC->values,
                                      (T)matC->ld,
                                      externalBuffer);
          break;
        }
      }
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
    case ALPHA_SPARSE_FORMAT_BLOCKED_ELL: {
      alphasparseSpMatDescr_t coo;
      alphasparseBell2coo<T, U>(matA, coo);
      spmm_template<T, U, V, W>(handle, opA, opB, alpha, coo, matB, beta, matC, alg, externalBuffer);
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
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
    return spmm_template<int32_t, double, double, double>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    return spmm_template<int32_t,
                         cuFloatComplex,
                         cuFloatComplex,
                         cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    return spmm_template<int32_t,
                         cuDoubleComplex,
                         cuDoubleComplex,
                         cuDoubleComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && matC->data_type == ALPHA_R_32I) {
    return spmm_template<int32_t, int8_t, int32_t, int32_t>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && matC->data_type == ALPHA_R_32F) {
    return spmm_template<int32_t, int8_t, float, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_32F) {
    return spmm_template<int32_t, half, float, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
      computeType == ALPHA_R_32F) {
    return spmm_template<int32_t, half, half, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F) {
    return spmm_template<int32_t, half2, half2, cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  }
#if (CUDA_ARCH >= 80)
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF &&
  //     computeType == ALPHA_R_32F) {
  //   return spmm_template<int32_t, nv_bfloat16, nv_bfloat16, float>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_C_16BF && matC->data_type == ALPHA_C_16BF &&
  //     computeType == ALPHA_C_32F) {
  //   return spmm_template<int32_t, nv_bfloat162, nv_bfloat162, cuFloatComplex>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, alg, externalBuffer);
  // }
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
	int block_size = 256;
	int warp_size =32;
	size_t typeSize = 4;
  if (matB->order == ALPHASPARSE_ORDER_ROW &&
      matC->order == ALPHASPARSE_ORDER_ROW) {
    switch (alg) {
      case ALPHASPARSE_SPMM_CSR_ALG1: {
        *bufferSize = 4;
        break;
      }
      default:
      case ALPHASPARSE_SPMM_CSR_ALG2: {
        int work_size;
        int N = matC->cols;
        if (N > 16) {
          work_size = 32;
        } else if (N > 8) {
          work_size = 16;
        } else if (N > 4) {
          work_size = 8;
        } else {
          work_size = 4;
        }
        *bufferSize = (CEIL(matA->nnz, work_size) + 1) * typeSize;
        break;
      }
      case ALPHASPARSE_SPMM_CSR_ALG3: {
        int work_size;
        int N = matC->cols;
        if (N > 16) {
          work_size = 32;
        } else if (N > 8) {
          work_size = 16;
        } else if (N > 4) {
          work_size = 8;
        } else {
          work_size = 4;
        }
        int total = matA->rows + matA->nnz;
        int worker_num = CEIL(total, work_size);
        *bufferSize = (worker_num + 1) * 2 * typeSize;
        break;
      }
      case ALPHASPARSE_SPMM_CSR_ALG4: {
        int work_size = CEIL(matA->nnz, matA->rows);
        int N = matC->cols;
        int subwarp_size;
        if (N > 16) {
          subwarp_size = 32;
        } else if (N > 8) {
          subwarp_size = 16;
        } else if (N > 4) {
          subwarp_size = 8;
        } else {
          subwarp_size = 4;
        }
        int block_num = CEIL(matA->nnz, block_size / subwarp_size * work_size);
        *bufferSize = (block_num + 1) * typeSize;
        break;
      }
      case ALPHASPARSE_SPMM_CSR_ALG5: {
        int work_size , subwarp_size;
        int N = matC->cols;
        if (N > 4) {
          subwarp_size = 8;
          work_size = 8;
        } else {
          subwarp_size = 4;
          work_size = 4;
        }
        *bufferSize = (CEIL(matA->nnz, block_size / subwarp_size * work_size) + 1) * typeSize;
        break;
      }
    }
  } else {
    switch (alg) {
      case ALPHASPARSE_SPMM_CSR_ALG2: {
        int work_size;
        int N = matC->cols;
        if (N > 4) {
          work_size = 8;
        } else {
          work_size = 4;
        }
        *bufferSize = (CEIL(matA->nnz, work_size) + 1) * typeSize;
        break;
      }
      case ALPHASPARSE_SPMM_CSR_ALG3: {
        int work_size;
        int N = matC->cols;
        if (N > 4) {
          work_size = 8;
        } else {
          work_size = 4;
        }
        int total = matA->rows + matA->nnz;
        int worker_num = CEIL(total, work_size);
        *bufferSize = (worker_num + 1) * 2 * typeSize;
        break;
      }
      case ALPHASPARSE_SPMM_CSR_ALG4: {
        int work_size = CEIL(matA->nnz, matA->rows);
        int N = matC->cols;
        int subwarp_size;
        if (N > 4) {
          subwarp_size = 8;
        } else {
          subwarp_size = 4;
        }
        int block_num = CEIL(matA->nnz, block_size / subwarp_size * work_size);
        *bufferSize = (block_num + 1) * typeSize;
        break;
      }
      case ALPHASPARSE_SPMM_CSR_ALG5: {
        int work_size , subwarp_size;
        int N = matC->cols;
        if (N > 4) {
          subwarp_size = 8;
          work_size = 8;
        } else {
          subwarp_size = 4;
          work_size = 4;
        }
        *bufferSize = (CEIL(matA->nnz, block_size / subwarp_size * work_size) + 1) * typeSize;
        break;
      }
      default: {
        *bufferSize = 4;
      }
    }
	}
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
