#include "../../format/transpose_csr.h"
#include "alphasparse.h"
#include "alphasparse/types.h" 
// #include "alphasparse_spmv_coo.h"
#include "alphasparse_spmv_csc.h"
#include "alphasparse_spmv_csr_scalar.h"
#include "alphasparse_spmv_csr_vector.h"
#include "alphasparse_spmv_csr_row_partition.h"
#include "alphasparse_spmv_csr_merge_ginkgo.h"
#include "alphasparse_spmv_csr_merge.h"
#include "alphasparse_spmv_csr_row_partition_ginkgo.h"
// #include "alphasparse_spmv_csr_row_partition_text.h"
#include "alphasparse_spmv_csr_line_enhance.h"
#include "alphasparse_spmv_csr_adaptive.h"
#include "alphasparse_spmv_csr_adaptive2.h"
#include "alphasparse_spmv_coo_row_partition_ginkgo.h"
#include "alphasparse_spmv_csr_flat.h"
#include "alphasparse_spmv_csr_line_enhance.h"
#include <iostream>

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmv_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              const void *alpha,
              alphasparseSpMatDescr_t matA,
              alphasparseDnVecDescr_t vecX,
              const void *beta,
              alphasparseDnVecDescr_t vecY,
              alphasparseSpMVAlg_t alg,
              void *externalBuffer)
{
  switch (matA->format)
  {
  case ALPHA_SPARSE_FORMAT_COO:
  {
    spmv_coo_ginkgo<T, U, V, W>(handle,
                                (T)matA->rows,
                                (T)matA->cols,
                                (T)matA->nnz,
                                *((W *)alpha),
                                (U *)matA->val_data,
                                (T *)matA->row_data,
                                (T *)matA->col_data,
                                (U *)vecX->values,
                                *((W *)beta),
                                (V *)vecY->values);
    break;
  }
  case ALPHA_SPARSE_FORMAT_CSR:
  {
    if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
    {
      transpose_csr<T, U>(matA);
    }
    switch (alg)
    {
    case ALPHA_SPARSE_SPMV_ALG_SCALAR:
    {
      spmv_csr_scalar<T, U, V, W>(handle,
                                  (T)matA->rows,
                                  (T)matA->cols,
                                  (T)matA->nnz,
                                  *((W *)alpha),
                                  (U *)matA->val_data,
                                  (T *)matA->row_data,
                                  (T *)matA->col_data,
                                  (U *)vecX->values,
                                  *((W *)beta),
                                  (V *)vecY->values);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_VECTOR:
    {
      spmv_csr_vector<T, U, V, W>(handle,
                                  (T)matA->rows,
                                  (T)matA->cols,
                                  (T)matA->nnz,
                                  *((W *)alpha),
                                  (U *)matA->val_data,
                                  (T *)matA->row_data,
                                  (T *)matA->col_data,
                                  (U *)vecX->values,
                                  *((W *)beta),
                                  (V *)vecY->values);
      break;
    }
    case ALPHA_SPARSE_SPMV_ROW_PARTITION:
    {
      spmv_csr_load<T, U, V, W>(handle,
                              (T)matA->rows,
                              (T)matA->cols,
                              (T)matA->nnz,
                              *((W *)alpha),
                              (U *)matA->val_data,
                              (T *)matA->row_data,
                              (T *)matA->col_data,
                              (U *)vecX->values,
                              *((W *)beta),
                              (V *)vecY->values,
                              externalBuffer);      
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_MERGE:
    {
      spmv_csr_merge_ginkgo<T, U, V, W>(handle,
                                        (T)matA->rows,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W *)alpha),
                                        (U *)matA->val_data,
                                        (T *)matA->row_data,
                                        (T *)matA->col_data,
                                        (U *)vecX->values,
                                        *((W *)beta),
                                        (V *)vecY->values,
                                        externalBuffer);
      break;
    }
    case ALPHA_SPARSE_SPMV_ADAPTIVE:
    {
      spmv_csr_adaptive2<T, U, V, W>(handle,
                                     (T)matA->rows,
                                     (T)matA->cols,
                                     (T)matA->nnz,
                                     *((W *)alpha),
                                     (U *)matA->val_data,
                                     (T *)matA->row_data,
                                     (T *)matA->col_data,
                                     (U *)vecX->values,
                                     *((W *)beta),
                                     (V *)vecY->values);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_LINE:
    {
      spmv_csr_line_adaptive<T, U, V, W>(handle,
                                      (T)matA->rows,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W *)alpha),
                                      (U *)matA->val_data,
                                      (T *)matA->row_data,
                                      (T *)matA->col_data,
                                      (U *)vecX->values,
                                      *((W *)beta),
                                      (V *)vecY->values,
                                      externalBuffer);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_FLAT1:
    {
      spmv_csr_flat<T, U, V, W, 1>(handle,
                                      (T)matA->rows,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W *)alpha),
                                      (U *)matA->val_data,
                                      (T *)matA->row_data,
                                      (T *)matA->col_data,
                                      (U *)vecX->values,
                                      *((W *)beta),
                                      (V *)vecY->values,
                                      externalBuffer);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_FLAT4:
    {
      spmv_csr_flat<T, U, V, W, 4>(handle,
                                      (T)matA->rows,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W *)alpha),
                                      (U *)matA->val_data,
                                      (T *)matA->row_data,
                                      (T *)matA->col_data,
                                      (U *)vecX->values,
                                      *((W *)beta),
                                      (V *)vecY->values,
                                      externalBuffer);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_FLAT8:
    {
      spmv_csr_flat<T, U, V, W, 8>(handle,
                                      (T)matA->rows,
                                      (T)matA->cols,
                                      (T)matA->nnz,
                                      *((W *)alpha),
                                      (U *)matA->val_data,
                                      (T *)matA->row_data,
                                      (T *)matA->col_data,
                                      (U *)vecX->values,
                                      *((W *)beta),
                                      (V *)vecY->values,
                                      externalBuffer);
      break;
    }
    }
    break;
  }
  case ALPHA_SPARSE_FORMAT_CSC:
  {
    spmv_csc<T, U, V, W>(handle,
                         (T)matA->rows,
                         (T)matA->cols,
                         (T)matA->nnz,
                         *((W *)alpha),
                         (U *)matA->val_data,
                         (T *)matA->row_data,
                         (T *)matA->col_data,
                         (U *)vecX->values,
                         *((W *)beta),
                         (V *)vecY->values);
    break;
  }
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpMV(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                const void *alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseDnVecDescr_t vecX,
                const void *beta,
                alphasparseDnVecDescr_t vecY,
                alphasparseDataType computeType,
                alphasparseSpMVAlg_t alg,
                void *externalBuffer)
{
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && vecY->data_type == ALPHA_R_32F)
  {
    return spmv_template<int32_t, float, float, float>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && vecY->data_type == ALPHA_R_64F)
  {
    return spmv_template<int32_t, double, double, double>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && vecY->data_type == ALPHA_C_32F)
  {
    return spmv_template<int32_t,
                         cuFloatComplex,
                         cuFloatComplex,
                         cuFloatComplex>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && vecY->data_type == ALPHA_C_64F)
  {
    return spmv_template<int32_t,
                         cuDoubleComplex,
                         cuDoubleComplex,
                         cuDoubleComplex>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_16F && vecY->data_type == ALPHA_R_16F)
  // {
  //   return spmv_template<int32_t, half, half, float>(
  //       handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_C_16F && vecY->data_type == ALPHA_C_16F)
  // {
  //   return spmv_template<int32_t, half2, half2, cuFloatComplex>(
  //       handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && vecY->data_type == ALPHA_R_32I)
  {
    return spmv_template<int32_t, int8_t, int32_t, int32_t>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && vecY->data_type == ALPHA_R_32F)
  {
    return spmv_template<int32_t, int8_t, float, float>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && vecY->data_type == ALPHA_R_32F)
  {
    return spmv_template<int32_t, half, float, float>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
#if (CUDA_ARCH >= 80)
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16BF && vecY->data_type == ALPHA_R_16BF)
  {
    return spmv_template<int32_t, nv_bfloat16, nv_bfloat16, float>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16BF && vecY->data_type == ALPHA_C_16BF)
  {
    return spmv_template<int32_t, nv_bfloat162, nv_bfloat162, cuFloatComplex>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
#endif
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpMV_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           const void *alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnVecDescr_t vecX,
                           const void *beta,
                           alphasparseDnVecDescr_t vecY,
                           alphasparseDataType computeType,
                           alphasparseSpMVAlg_t alg,
                           size_t *bufferSize)
{
  switch (alg)
  {
  case ALPHA_SPARSE_SPMV_ALG_MERGE:
  {
    const int total = matA->rows + matA->nnz;
    const int block_num =
        ceildivT(total, SPMV_MERGE_BLOCK_SIZE * 8);
    size_t typeSize = 4;
    if (computeType == ALPHA_R_32F)
      typeSize = 4;
    else if (computeType == ALPHA_R_64F)
      typeSize = 8;
    else if (computeType == ALPHA_C_32F)
      typeSize = 8;
    else if (computeType == ALPHA_C_64F)
      typeSize = 16;
    *bufferSize = (block_num * 2 + 2) * typeSize;
    break;
  }
  case ALPHA_SPARSE_SPMV_ROW_PARTITION:
  {
    const int SM = 520;
    const int MAX_WARP_PER_SM = 64;
    const int64_t warp_size = 32;
    const int64_t nwarps_ = SM * MAX_WARP_PER_SM / warp_size;
    int nwarps = clac_size(matA->nnz, warp_size, nwarps_);
    size_t typeSize = 4;
    if (computeType == ALPHA_R_32F)
      typeSize = 4;
    else if (computeType == ALPHA_R_64F)
      typeSize = 8;
    else if (computeType == ALPHA_C_32F)
      typeSize = 8;
    else if (computeType == ALPHA_C_64F)
      typeSize = 16;
    *bufferSize = nwarps * typeSize;
    break;
  }
  case ALPHA_SPARSE_SPMV_ALG_FLAT1:
  {
    constexpr int R = 2;
    const int BLOCK_SIZE = 512;
    const int nnz_per_block = R * BLOCK_SIZE;
    const int nwarps = ceildivT<int>(matA->nnz, nnz_per_block);
    *bufferSize = (nwarps + 1) * sizeof(int);
    break;
  }
  case ALPHA_SPARSE_SPMV_ALG_FLAT4:
  {
    constexpr int R = 2;
    const int BLOCK_SIZE = 512;
    const int nnz_per_block = R * BLOCK_SIZE;
    const int nwarps = ceildivT<int>(matA->nnz, nnz_per_block);
    *bufferSize = (nwarps + 1) * sizeof(int);
    break;
  }
  case ALPHA_SPARSE_SPMV_ALG_FLAT8:
  {
    constexpr int R = 2;
    const int BLOCK_SIZE = 512;
    const int nnz_per_block = R * BLOCK_SIZE;
    const int nwarps = ceildivT<int>(matA->nnz, nnz_per_block);
    *bufferSize = (nwarps + 1) * sizeof(int);
    break;
  }
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
