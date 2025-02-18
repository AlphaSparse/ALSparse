
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
#include "../../format/csr2csc.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;

int m, n, nnz;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
half* coo_values;

// coo format
half* x_val;
float* ict_y;
float* cuda_y;

// parms for kernel
const float alpha = 2.f;
const float beta = 3.f;

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
cuda_mv()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  half* dX = NULL;
  float* dY = NULL;
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  half* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);
  cusparseDnVecDescr_t vecX, vecY;
  cusparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(half)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(float), cudaMemcpyHostToDevice));
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, dX, CUDA_R_16F));
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, dY, CUDA_R_32F));
  int* dCscColPtr = NULL;
  int* dCscRowInd = NULL;
  half* dCscVal = NULL;
  size_t csc_bufferSize = 0;
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCscRowInd, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCscVal, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCscColPtr, sizeof(int) * (n + 1)));
  cusparseCsr2cscEx2_bufferSize(handle,
    m,
    n,
    nnz,
    dAval,
    dCsrRowPtr,
    dAcol,
    dCscVal,
    dCscColPtr,
    dCscRowInd,
    CUDA_R_16F,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    &csc_bufferSize);
  void* csc_dBuffer = NULL;
  CHECK_CUDA(cudaMalloc((void**)&csc_dBuffer, csc_bufferSize));
  cusparseCsr2cscEx2(handle,
    m,
    n,
    nnz,
    dAval,
    dCsrRowPtr,
    dAcol,
    dCscVal,
    dCscColPtr,
    dCscRowInd,
    CUDA_R_16F,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    csc_dBuffer);

  cusparseSpMatDescr_t csc;
  cusparseCreateCsc(&csc,
    m,
    n,
    nnz,
    dCscColPtr,
    dCscRowInd,
    dCscVal,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_16F);
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         csc,
                                         vecX,
                                         &beta,
                                         vecY,
                                         CUDA_R_32F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         &bufferSize));
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(cusparseSpMV(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              csc,
                              vecX,
                              &beta,
                              vecY,
                              CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT,
                              dBuffer));
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dX);
  cudaFree(dY);
  cusparseDestroy(handle);
}

static void
alpha_mv()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  half* dX = NULL;
  float* dY = NULL;
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  half* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));

  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);
  alphasparseDnVecDescr_t vecX, vecY;
  alphasparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(half)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseDnVecDescr_t x{};
  alphasparseCreateDnVec(&x, n, (void*)dX, ALPHA_R_16F);

  alphasparseDnVecDescr_t y_ict{};
  alphasparseCreateDnVec(&y_ict, m, (void*)dY, ALPHA_R_32F);

  alphasparseSpMatDescr_t csr;
  alphasparseCreateCsr(&csr,
                       m,
                       n,
                       nnz,
                       dCsrRowPtr,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_16F);
  alphasparseSpMatDescr_t csc;
  alphasparseCsr2csc<int, half>(csr, csc);
  
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  alphasparseSpMV_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             csc,
                             x,
                             &beta,
                             y_ict,
                             ALPHA_R_32F,
                             ALPHA_SPARSE_SPMV_ALG_DEFAULT,
                             &bufferSize);
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  alphasparseSpMV(handle,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha,
                  csc,
                  x,
                  &beta,
                  y_ict,
                  ALPHA_R_32F,
                  ALPHA_SPARSE_SPMV_ALG_DEFAULT,
                  dBuffer);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(ict_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
}

int
main(int argc, const char* argv[])
{
  // args
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);

  // read coo
  alpha_read_coo<half>(
    file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, half>(nnz, coo_row_index, coo_col_index, coo_values);
  csrRowPtr = (int*)alpha_malloc(sizeof(int) * (m + 1));
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    int temp = n;
    n = m;
    m = temp;
  }
  // for (int i = 0; i < 20; i++) {
  //   std::cout << coo_row_index[i] << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << coo_col_index[i] << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << coo_values[i] << ", ";
  // }
  // std::cout << std::endl;
  // init x y
  x_val = (half*)alpha_malloc(n * sizeof(half));
  ict_y = (float*)alpha_malloc(m * sizeof(float));
  cuda_y = (float*)alpha_malloc(m * sizeof(float));
  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  for (int i = 0; i < 20; i++) {
    std::cout << cuda_y[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << ict_y[i] << ", ";
  }
  cuda_mv();
  alpha_mv();
  check((float*)cuda_y, m, (float*)ict_y, m);
  for (int i = 0; i < 20; i++) {
    std::cout << cuda_y[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << ict_y[i] << ", ";
  }
  return 0;
}
