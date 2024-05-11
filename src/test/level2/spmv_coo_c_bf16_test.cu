
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
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;

int m, n, nnz;
int *coo_row_index, *coo_col_index;
nv_bfloat162* coo_values;

// coo format
nv_bfloat162* x_val;
nv_bfloat162* ict_y;
nv_bfloat162* cuda_y;

// parms for kernel
const cuFloatComplex alpha = {1.f,2.f};
const cuFloatComplex beta = {3.f,4.f};

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
  nv_bfloat162* dX = NULL;
  nv_bfloat162* dY = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  nv_bfloat162* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(nv_bfloat162) * nnz));
  CHECK_CUDA(cudaMemcpy(dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAval, coo_values, nnz * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));
  cusparseDnVecDescr_t vecX, vecY;
  cusparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(nv_bfloat162)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(nv_bfloat162)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, dX, CUDA_C_16BF));
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, dY, CUDA_C_16BF));
  CHECK_CUSPARSE(cusparseCreateCoo(&matA,
                                   m,
                                   n,
                                   nnz,
                                   dArow,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_C_16BF));
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         vecX,
                                         &beta,
                                         vecY,
                                         CUDA_C_32F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         &bufferSize));
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(cusparseSpMV(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              matA,
                              vecX,
                              &beta,
                              vecY,
                              CUDA_C_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT,
                              dBuffer));
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(nv_bfloat162) * m, cudaMemcpyDeviceToHost));
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
  nv_bfloat162* dX = NULL;
  nv_bfloat162* dY = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  nv_bfloat162* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(nv_bfloat162) * nnz));
  CHECK_CUDA(cudaMemcpy(dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAval, coo_values, nnz * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));
  alphasparseDnVecDescr_t vecX, vecY;
  alphasparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(nv_bfloat162)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(nv_bfloat162)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(nv_bfloat162), cudaMemcpyHostToDevice));

  alphasparseDnVecDescr_t x{};
  alphasparseCreateDnVec(&x, n, (void*)dX, ALPHA_C_16BF);

  alphasparseDnVecDescr_t y_ict{};
  alphasparseCreateDnVec(&y_ict, m, (void*)dY, ALPHA_C_16BF);

  alphasparseSpMatDescr_t coo;
  alphasparseCreateCoo(&coo,
                       m,
                       n,
                       nnz,
                       dArow,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_C_16BF);
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  alphasparseSpMV_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             coo,
                             x,
                             &beta,
                             y_ict,
                             ALPHA_C_32F,
                             ALPHA_SPARSE_SPMV_ALG_COO,
                             &bufferSize);
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  alphasparseSpMV(handle,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha,
                  coo,
                  x,
                  &beta,
                  y_ict,
                  ALPHA_C_32F,
                  ALPHA_SPARSE_SPMV_ALG_COO,
                  dBuffer);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(ict_y, dY, sizeof(nv_bfloat162) * m, cudaMemcpyDeviceToHost));
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
  alpha_read_coo<nv_bfloat162>(
    file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, nv_bfloat162>(nnz, coo_row_index, coo_col_index, coo_values);
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
  x_val = (nv_bfloat162*)alpha_malloc(n * sizeof(nv_bfloat162));
  ict_y = (nv_bfloat162*)alpha_malloc(m * sizeof(nv_bfloat162));
  cuda_y = (nv_bfloat162*)alpha_malloc(m * sizeof(nv_bfloat162));

  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  cuda_mv();
  alpha_mv();
  check((nv_bfloat162*)cuda_y, m, (nv_bfloat162*)ict_y, m);
  for(int i=0;i<20;i++){
    std::cout<<cuda_y[i]<<", ";
  }
  std::cout<<std::endl;
  for(int i=0;i<20;i++){
    std::cout<<ict_y[i]<<", ";
  }
  return 0;
}
