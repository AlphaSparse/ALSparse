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

#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
int thread_num;
bool check_flag;
int iter;

cusparseSpMMAlg_t cu_alg = CUSPARSE_SPMM_ALG_DEFAULT;
alphasparseSpMMAlg_t alpha_alg = ALPHASPARSE_SPMM_CSR_ALG1;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
cuDoubleComplex* coo_values;

// parms for kernel
cuDoubleComplex *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
int B_rows;
long long ldb, ldc;
long long B_size, C_size;
const cuDoubleComplex alpha = {2.f, 3.f};
const cuDoubleComplex beta = {3.f, 2.f};

cudaEvent_t event_start, event_stop;
float elapsed_time = 0.0;

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             cudaGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
             __LINE__,                                                         \
             cusparseGetErrorString(status),                                   \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

static void
cuda_mm()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  cuDoubleComplex* dAval = NULL;

  int nnz = rnnz;

  cuDoubleComplex* dmatB = NULL;
  cuDoubleComplex* dmatC = NULL;

  cudaMalloc((void**)&dmatB, sizeof(cuDoubleComplex) * B_size);
  cudaMalloc((void**)&dmatC, sizeof(cuDoubleComplex) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(cuDoubleComplex) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  cudaMemcpy(dmatB, hmatB, sizeof(cuDoubleComplex) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_roc, sizeof(cuDoubleComplex) * C_size, cudaMemcpyHostToDevice);
  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(cusparseCreateDnMat(
    &matB, A_cols, C_cols, ldb, dmatB, CUDA_C_64F, CUSPARSE_ORDER_COL))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, CUDA_C_64F, CUSPARSE_ORDER_COL))
  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dCsrRowPtr,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_C_64F));
  std::vector<double> times;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matB,
                                         &beta,
                                         matC,
                                         CUDA_C_64F,
                                         cu_alg,
                                         &bufferSize))
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(cusparseSpMM(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              matA,
                              matB,
                              &beta,
                              matC,
                              CUDA_C_64F,
                              cu_alg,
                              dBuffer))
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  times.push_back(elapsed_time);
  printf("cusparse: %lf\n", get_avg_time(times));
  CHECK_CUDA(
    cudaMemcpy(matC_roc, dmatC, C_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))
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
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  cuDoubleComplex* dAval = NULL;

  int nnz = rnnz;

  cuDoubleComplex* dmatB = NULL;
  cuDoubleComplex* dmatC = NULL;

  cudaMalloc((void**)&dmatB, sizeof(cuDoubleComplex) * B_size);
  cudaMalloc((void**)&dmatC, sizeof(cuDoubleComplex) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(cuDoubleComplex) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  cudaMemcpy(dmatB, hmatB, sizeof(cuDoubleComplex) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_ict, sizeof(cuDoubleComplex) * C_size, cudaMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
    &matB, A_cols, C_cols, ldb, dmatB, ALPHA_C_64F, ALPHASPARSE_ORDER_COL);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_C_64F, ALPHASPARSE_ORDER_COL);
  alphasparseSpMatDescr_t matA;
  alphasparseCreateCsr(&matA,
                       A_rows,
                       A_cols,
                       nnz,
                       dCsrRowPtr,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_C_64F);
  std::vector<double> times;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  size_t bufferSize = 0;
  alphasparseSpMM_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             matA,
                             matB,
                             &beta,
                             matC,
                             ALPHA_C_64F,
                             alpha_alg,
                             &bufferSize);
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  alphasparseSpMM(handle,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha,
                  matA,
                  matB,
                  &beta,
                  matC,
                  ALPHA_C_64F,
                  alpha_alg,
                  dBuffer);
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  times.push_back(elapsed_time);
  printf("alphasparse: %lf\n", get_avg_time(times));
  CHECK_CUDA(
    cudaMemcpy(matC_ict, dmatC, C_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dmatB);
  cudaFree(dmatC);
}

int
main(int argc, const char* argv[])
{
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);

  // read coo
  alpha_read_coo<cuDoubleComplex>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, cuDoubleComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, 256);
  C_rows = A_rows;
  C_cols = columns;
  B_rows = A_cols;
  ldb = B_rows;
  ldc = C_rows;
  B_size = ldb * C_cols;
  C_size = ldc * C_cols;
  // init x y
  // init B C
  hmatB = (cuDoubleComplex*)alpha_malloc(B_size * sizeof(cuDoubleComplex));
  matC_ict = (cuDoubleComplex*)alpha_malloc(C_size * sizeof(cuDoubleComplex));
  matC_roc = (cuDoubleComplex*)alpha_malloc(C_size * sizeof(cuDoubleComplex));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  alpha_fill_random(matC_roc, 1, C_size);
  // std::cout << "matc:" <<std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << matC_ict[i] << ", ";
  // }
  // std::cout << "matc:" << std::endl;
  cuda_mm();
  alpha_mm();

  for (int i = 0; i < 8; i++) {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 8; i++) {
    std::cout << matC_ict[i] << ", ";
  }
  check((cuDoubleComplex*)matC_roc, C_size, (cuDoubleComplex*)matC_ict, C_size);
  return 0;
}
