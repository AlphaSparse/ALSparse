
#include "../test_common.h"

/**
 * @brief ict dcu mv hyb test
 * @author HPCRC, ICT
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_coo.h"
#include "../../format/coo2bell.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char *file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
int8_t *coo_values;

// parms for kernel
int8_t *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
long long B_rows, B_cols;
long long ldb, ldc;
long long B_size, C_size;
const int8_t alpha = 2;
const int8_t beta = 3;
int blocksize;

#define CHECK_CUDA(func)                                         \
  {                                                              \
    cudaError_t status = (func);                                 \
    if (status != cudaSuccess)                                   \
    {                                                            \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                           \
             cudaGetErrorString(status),                         \
             status);                                            \
      exit(-1);                                                  \
    }                                                            \
  }

#define CHECK_CUSPARSE(func)                                         \
  {                                                                  \
    cusparseStatus_t status = (func);                                \
    if (status != CUSPARSE_STATUS_SUCCESS)                           \
    {                                                                \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                               \
             cusparseGetErrorString(status),                         \
             status);                                                \
      exit(-1);                                                      \
    }                                                                \
  }

static void
cuda_mm()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  int *dArow = NULL;
  int *dAcol = NULL;
  int8_t *dAval = NULL;

  int nnz = rnnz;

  int8_t *dmatB = NULL;
  int8_t *dmatC = NULL;

  cudaMalloc((void **)&dmatB, sizeof(int8_t) * B_size);
  cudaMalloc((void **)&dmatC, sizeof(int8_t) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAval, sizeof(int8_t) * nnz));

  CHECK_CUDA(cudaMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dAval, coo_values, nnz * sizeof(int8_t), cudaMemcpyHostToDevice));

  cudaMemcpy(dmatB, hmatB, sizeof(int8_t) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_roc, sizeof(int8_t) * C_size, cudaMemcpyHostToDevice);
  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(cusparseCreateDnMat(
      &matB, A_cols, B_cols, ldb, dmatB, CUDA_R_8I, CUSPARSE_ORDER_ROW))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(
      &matC, C_rows, C_cols, ldc, dmatC, CUDA_R_8I, CUSPARSE_ORDER_ROW))
  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateCoo(&matA,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dArow,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_8I));
  alphasparseSpMatDescr_t matA_ict;
  alphasparseCreateCoo(&matA_ict,
                       A_rows,
                       A_cols,
                       nnz,
                       dArow,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_8I);
  alphasparseSpMatDescr_t bell_ict;
  alphasparseCoo2bell<int, int8_t>(matA_ict, bell_ict, blocksize);
  cusparseSpMatDescr_t bell;
  cusparseCreateBlockedEll(&bell,
                           A_rows,
                           A_cols,
                           blocksize,
                           bell_ict->ell_cols,
                           bell_ict->col_data,
                           bell_ict->val_data,
                           CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_8I);
  size_t bufferSize = 0;
  cusparseOperation_t cutransA, cutransB;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
    cutransA = CUSPARSE_OPERATION_TRANSPOSE;
  else if (transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    cutransA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else
    cutransA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  if (transB == ALPHA_SPARSE_OPERATION_TRANSPOSE)
    cutransB = CUSPARSE_OPERATION_TRANSPOSE;
  else if (transB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    cutransB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else
    cutransB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                         cutransA,
                                         cutransB,
                                         &alpha,
                                         bell,
                                         matB,
                                         &beta,
                                         matC,
                                         CUDA_R_32I,
                                         CUSPARSE_SPMM_ALG_DEFAULT,
                                         &bufferSize))
  void *dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  printf(" CUDA TRANS %d\n", cutransA);
  CHECK_CUSPARSE(cusparseSpMM(handle,
                              cutransA,
                              cutransB,
                              &alpha,
                              bell,
                              matB,
                              &beta,
                              matC,
                              CUDA_R_32I,
                              CUSPARSE_SPMM_ALG_DEFAULT,
                              dBuffer))
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(matC_roc, dmatC, C_size * sizeof(int8_t), cudaMemcpyDeviceToHost))
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dmatB);
  cudaFree(dmatC);
  cusparseDestroy(handle);
}

static void
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  int *dArow = NULL;
  int *dAcol = NULL;
  int8_t *dAval = NULL;

  int nnz = rnnz;

  int8_t *dmatB = NULL;
  int8_t *dmatC = NULL;

  cudaMalloc((void **)&dmatB, sizeof(int8_t) * B_size);
  cudaMalloc((void **)&dmatC, sizeof(int8_t) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAval, sizeof(int8_t) * nnz));

  CHECK_CUDA(cudaMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dAval, coo_values, nnz * sizeof(int8_t), cudaMemcpyHostToDevice));

  cudaMemcpy(dmatB, hmatB, sizeof(int8_t) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_ict, sizeof(int8_t) * C_size, cudaMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
      &matB, A_cols, B_cols, ldb, dmatB, ALPHA_R_8I, ALPHASPARSE_ORDER_ROW);
  // Create dense matrix C
  alphasparseCreateDnMat(
      &matC, C_rows, C_cols, ldc, dmatC, ALPHA_R_8I, ALPHASPARSE_ORDER_ROW);
  alphasparseSpMatDescr_t matA;
  alphasparseCreateCoo(&matA,
                       A_rows,
                       A_cols,
                       nnz,
                       dArow,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_8I);
  alphasparseSpMatDescr_t bell;
  alphasparseCoo2bell<int, int8_t>(matA, bell, blocksize);
  size_t bufferSize = 0;
  alphasparseSpMM_bufferSize(handle,
                             transA,
                             transB,
                             &alpha,
                             bell,
                             matB,
                             &beta,
                             matC,
                             ALPHA_R_32I,
                             ALPHASPARSE_SPMM_ALG_DEFAULT,
                             &bufferSize);
  void *dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  printf(" ALPHA TRANS %d\n", transA);
  alphasparseSpMM(handle,
                  transA,
                  transB,
                  &alpha,
                  bell,
                  matB,
                  &beta,
                  matC,
                  ALPHA_R_32I,
                  ALPHASPARSE_SPMM_ALG_DEFAULT,
                  dBuffer);
  CHECK_CUDA(
      cudaMemcpy(matC_ict, dmatC, C_size * sizeof(int8_t), cudaMemcpyDeviceToHost))
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dmatB);
  cudaFree(dmatC);
}

int main(int argc, const char *argv[])
{
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);
  blocksize = args_get_blocksize(argc, argv);

  // read coo
  alpha_read_coo<int8_t>(
      file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, int8_t>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, A_cols); // 默认C是方阵

  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
  {
    if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    {
      C_rows = A_rows;
      C_cols = columns;
      B_cols = columns;
      ldb = A_cols;
      ldc = A_rows;
    }
    else
    {
      C_rows = A_rows;
      C_cols = columns;
      B_cols = A_cols;
      ldb = columns;
      ldc = A_rows;
    }
  }
  else // transA, conjA
  {
    if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    {
      C_rows = A_cols;
      C_cols = columns;
      B_cols = columns;
      ldb = A_rows;
      ldc = C_rows;
    }
    else // transB, conjB, B转置就用方阵测
    {
      C_rows = A_rows;
      C_cols = columns;
      B_cols = A_rows;
      ldb = columns;
      ldc = C_rows;
    }
  }

  B_size = ldb * B_cols;
  C_size = ldc * C_cols;
  for (int i = 0; i < 20; i++)
  {
    std::cout << coo_row_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++)
  {
    std::cout << coo_col_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++)
  {
    std::cout << coo_values[i] << ", ";
  }
  std::cout << std::endl;
  // init x y
  // init B C
  hmatB = (int8_t *)alpha_malloc(B_size * sizeof(int8_t));
  matC_ict = (int8_t *)alpha_malloc(C_size * sizeof(int8_t));
  matC_roc = (int8_t *)alpha_malloc(C_size * sizeof(int8_t));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  alpha_fill_random(matC_roc, 1, C_size);

  cuda_mm();
  alpha_mm();

  for (int i = 0; i < 40; i++)
  {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 40; i++)
  {
    std::cout << matC_ict[i] << ", ";
  }
  std::cout << std::endl;
  check((int8_t *)matC_roc, C_size, (int8_t *)matC_ict, C_size);
  return 0;
}
