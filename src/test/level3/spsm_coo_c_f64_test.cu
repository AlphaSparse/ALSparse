
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
alphasparseOperation_t transB;
alpha_matrix_descr descrT;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
cuDoubleComplex* coo_values;

// parms for kernel
cuDoubleComplex *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
long long B_cols;
long long ldb, ldc;
long long B_size, C_size;
const cuDoubleComplex alpha = {2.f, 3.f};

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
  cusparseSpSMDescr_t spsmDescr;
  CHECK_CUSPARSE(cusparseSpSM_createDescr(&spsmDescr))
  // Offload data to device
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
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  cudaMemcpy(dmatB, hmatB, sizeof(cuDoubleComplex) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_roc, sizeof(cuDoubleComplex) * C_size, cudaMemcpyHostToDevice);
  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(cusparseCreateDnMat(
    &matB, A_cols, B_cols, ldb, dmatB, CUDA_C_64F, CUSPARSE_ORDER_COL))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, CUDA_C_64F, CUSPARSE_ORDER_COL))
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
                                   CUDA_C_64F));

  cusparseFillMode_t fillmode;
  cusparseDiagType_t diagtype;
  if (descrT.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    fillmode = CUSPARSE_FILL_MODE_LOWER;
  else
    fillmode = CUSPARSE_FILL_MODE_UPPER;
  if (descrT.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
    diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  else
    diagtype = CUSPARSE_DIAG_TYPE_UNIT;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
    matA, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode)))
  // Specify Unit|Non-Unit diagonal type.
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
    matA, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype)))
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpSM_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matB,
                                         matC,
                                         CUDA_C_64F,
                                         CUSPARSE_SPSM_ALG_DEFAULT,
                                         spsmDescr,
                                         &bufferSize))
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(cusparseSpSM_analysis(handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha,
                                       matA,
                                       matB,
                                       matC,
                                       CUDA_C_64F,
                                       CUSPARSE_SPSM_ALG_DEFAULT,
                                       spsmDescr,
                                       dBuffer))
  CHECK_CUSPARSE(cusparseSpSM_solve(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    matC,
                                    CUDA_C_64F,
                                    CUSPARSE_SPSM_ALG_DEFAULT,
                                    spsmDescr))
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
  alphasparseSpSMDescr_t spsmDescr;
  alphasparseSpSM_createDescr(&spsmDescr);
  // Offload data to device
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
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  cudaMemcpy(dmatB, hmatB, sizeof(cuDoubleComplex) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_ict, sizeof(cuDoubleComplex) * C_size, cudaMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
    &matB, A_cols, B_cols, ldb, dmatB, ALPHA_C_64F, ALPHASPARSE_ORDER_COL);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_C_64F, ALPHASPARSE_ORDER_COL);
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
                       ALPHA_C_64F);
  // Specify Lower|Upper fill mode.
  alphasparseSpMatSetAttribute(
    matA, ALPHASPARSE_SPMAT_FILL_MODE, &descrT.mode, sizeof(descrT.mode));
  // Specify Unit|Non-Unit diagonal type.
  alphasparseSpMatSetAttribute(
    matA, ALPHASPARSE_SPMAT_DIAG_TYPE, &descrT.diag, sizeof(descrT.diag));
  size_t bufferSize = 0;
  alphasparseSpSM_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             matA,
                             matB,
                             matC,
                             ALPHA_C_64F,
                             ALPHASPARSE_SPSM_ALG_DEFAULT,
                             spsmDescr,
                             &bufferSize);
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  alphasparseSpSM_solve(handle,
                        ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                        ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        matA,
                        matB,
                        matC,
                        ALPHA_C_64F,
                        ALPHASPARSE_SPSM_ALG_DEFAULT,
                        spsmDescr);
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
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);
  descrT = alpha_args_get_matrix_descrA(argc, argv);

  // read coo
  alpha_read_coo<cuDoubleComplex>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, cuDoubleComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = 1024; // 默认C是方阵
  C_rows = A_rows;
  C_cols = columns;
  B_cols = columns;
  ldb = A_cols;
  ldc = C_rows;
  B_size = ldb * B_cols;
  C_size = ldc * B_cols;
  for (int i = 0; i < 20; i++) {
    std::cout << coo_row_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << coo_col_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << coo_values[i] << ", ";
  }
  std::cout << std::endl;
  // init x y
  // init B C
  hmatB = (cuDoubleComplex*)alpha_malloc(B_size * sizeof(cuDoubleComplex));
  matC_ict = (cuDoubleComplex*)alpha_malloc(C_size * sizeof(cuDoubleComplex));
  matC_roc = (cuDoubleComplex*)alpha_malloc(C_size * sizeof(cuDoubleComplex));
  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  alpha_fill_random(matC_roc, 1, C_size);
  cuda_mm();
  alpha_mm();

  for (int i = 0; i < 20; i++) {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << matC_ict[i] << ", ";
  }
  check((cuDoubleComplex*)matC_roc, C_size, (cuDoubleComplex*)matC_ict, C_size);
  return 0;
}
