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

const int iteration = 210;
const int warmup_times = 200;
const char *file, *metric_file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

double alpha_time;
std::vector<alphasparseSpMMAlg_t> alpha_alg_list = {ALPHASPARSE_SPMM_ALG_DEFAULT, ALPHASPARSE_SPMM_CSR_ALG1, ALPHASPARSE_SPMM_CSR_ALG2, ALPHASPARSE_SPMM_CSR_ALG3, ALPHASPARSE_SPMM_CSR_ALG4, ALPHASPARSE_SPMM_CSR_ALG5};
// std::vector<alphasparseSpMMAlg_t> alpha_alg_list = {ALPHASPARSE_SPMM_CSR_ALG2};
int alg_num;

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
    &matB, A_cols, C_cols, ldb, dmatB, ALPHA_C_64F, ALPHASPARSE_ORDER_ROW);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_C_64F, ALPHASPARSE_ORDER_ROW);
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
  auto alg = alpha_alg_list[alg_num];
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
                             alg,
                             &bufferSize);
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  std::vector<double> times;
  for (int i = 0; i < iteration; i++) {
    if ( i>= warmup_times) cudaMemcpy(dmatC, matC_ict, sizeof(cuDoubleComplex) * C_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    alphasparseSpMM(handle,
                    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    matA,
                    matB,
                    &beta,
                    matC,
                    ALPHA_C_64F,
                    alg,
                    dBuffer);
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    if (i >= warmup_times)
      times.push_back(elapsed_time);
  }
  alpha_time = get_avg_time_2(times);
  printf("alphasparse %d: %lf ms\n", alg, alpha_time);

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
	metric_file = args_save_metrics_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  alg_num = args_get_alg_num(argc, argv);

  // read coo
  alpha_read_coo<cuDoubleComplex>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, cuDoubleComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, 256);
  C_rows = A_rows;
  C_cols = columns;
  B_rows = A_cols;
  ldb = columns;
  ldc = C_cols;
  B_size = ldb * B_rows;
  C_size = ldc * C_rows;
  // init x y
  // init B C
  hmatB = (cuDoubleComplex*)alpha_malloc(B_size * sizeof(cuDoubleComplex));
  matC_ict = (cuDoubleComplex*)alpha_malloc(C_size * sizeof(cuDoubleComplex));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  // std::cout << "matc:" <<std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << matC_ict[i] << ", ";
  // }
  // std::cout << "matc:" << std::endl;
  alpha_mm();

  std::ofstream filename(metric_file, std::ios::app);
  filename << file << "," << alpha_time << "\n";
  filename.close();
  return 0;
}
