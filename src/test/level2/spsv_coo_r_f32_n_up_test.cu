
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
float* coo_values;

// coo format
float* x_val;
float* ict_y;
float* cuda_y;

// parms for kernel
const float alpha = 2.f;

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
cuda_spsv()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  float* dX = NULL;
  float* dY = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  float* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(float) * nnz));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
  cusparseDnVecDescr_t vecX, vecY;
  cusparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(float), cudaMemcpyHostToDevice));
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, dX, CUDA_R_32F));
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, dY, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateCoo(&matA,
                                   m,
                                   n,
                                   nnz,
                                   dArow,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
  cusparseSpSVDescr_t spsvDescr;
  cusparseSpSV_createDescr(&spsvDescr);
  cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
    matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode)))
  // Specify Unit|Non-Unit diagonal type.
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
    matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype)))
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  // allocate an external buffer for analysis
  CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         vecX,
                                         vecY,
                                         CUDA_R_32F,
                                         CUSPARSE_SPSV_ALG_DEFAULT,
                                         spsvDescr,
                                         &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  CHECK_CUSPARSE(cusparseSpSV_analysis(handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha,
                                       matA,
                                       vecX,
                                       vecY,
                                       CUDA_R_32F,
                                       CUSPARSE_SPSV_ALG_DEFAULT,
                                       spsvDescr,
                                       dBuffer))
  // execute SpSV
  CHECK_CUSPARSE(cusparseSpSV_solve(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    vecX,
                                    vecY,
                                    CUDA_R_32F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    spsvDescr))
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
  CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
  CHECK_CUSPARSE(cusparseDestroy(handle))
}

static void
alpha_spsv()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  float* dX = NULL;
  float* dY = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  float* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(float) * nnz));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseDnVecDescr_t x{};
  alphasparseCreateDnVec(&x, n, (void*)dX, ALPHA_R_32F);

  alphasparseDnVecDescr_t y_ict{};
  alphasparseCreateDnVec(&y_ict, m, (void*)dY, ALPHA_R_32F);

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
                       ALPHA_R_32F);
  alphasparseSpSVDescr_t spsvDescr;
  alphasparseSpSV_createDescr(&spsvDescr);
  // Specify Lower|Upper fill mode.
  alphasparse_fill_mode_t fillmode = ALPHA_SPARSE_FILL_MODE_UPPER;
  alphasparseSpMatSetAttribute(
    coo, ALPHASPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
  // Specify Unit|Non-Unit diagonal type.
  alphasparse_diag_type_t diagtype = ALPHA_SPARSE_DIAG_NON_UNIT;
  alphasparseSpMatSetAttribute(
    coo, ALPHASPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  alphasparseSpSV_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             coo,
                             x,
                             y_ict,
                             ALPHA_R_32F,
                             ALPHA_SPARSE_SPSV_ALG_DEFAULT,
                             spsvDescr,
                             &bufferSize);
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  alphasparseSpSV_analysis(handle,
                           ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha,
                           coo,
                           x,
                           y_ict,
                           ALPHA_R_32F,
                           ALPHA_SPARSE_SPSV_ALG_DEFAULT,
                           spsvDescr,
                           dBuffer);
  alphasparseSpSV_solve(handle,
                        ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        coo,
                        x,
                        y_ict,
                        ALPHA_R_32F,
                        ALPHA_SPARSE_SPSV_ALG_DEFAULT,
                        spsvDescr);
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
  alpha_read_coo<float>(
    file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, float>(nnz, coo_row_index, coo_col_index, coo_values);
  csrRowPtr = (int*)alpha_malloc(sizeof(int) * (m + 1));
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    int temp = n;
    n = m;
    m = temp;
  }
  // for (int i = 0; i < 3; i++) {
  //   std::cout << coo_row_index[i] << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 3; i++) {
  //   std::cout << coo_col_index[i] << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 3; i++) {
  //   std::cout << coo_values[i] << ", ";
  // }
  // std::cout << std::endl;
  // init x y
  x_val = (float*)alpha_malloc(n * sizeof(float));
  ict_y = (float*)alpha_malloc(m * sizeof(float));
  cuda_y = (float*)alpha_malloc(m * sizeof(float));

  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  cuda_spsv();
  alpha_spsv();
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
