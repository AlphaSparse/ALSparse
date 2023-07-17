
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

alphasparseOperation_t transA;
alphasparseOperation_t transB;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
half* coo_values;

// parms for kernel
half *hmatB, *matC_ict, *matC_roc;
long long ldb, ldc;
long long B_size, nnz, nnz_c;
const half alpha = 2.f;
const half beta = 3.f;

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

  int nnz = rnnz;

  // Offload data to device
  int* dACsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  half* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dACsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dACsrRowPtr);

  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dACsrRowPtr,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_16F));

  int* dBCsrRowPtr = NULL;
  int* dBrow = NULL;
  int* dBcol = NULL;
  half* dBval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBrow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dBCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dBrow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dBcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dBval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dBrow, nnz, A_rows, dBCsrRowPtr);

  cusparseSpMatDescr_t matB;
  CHECK_CUSPARSE(cusparseCreateCsr(&matB,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dBCsrRowPtr,
                                   dBcol,
                                   dBval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_16F));

  int* dCCsrRowPtr = NULL;
  int* dCrow = NULL;
  int* dCcol = NULL;
  half* dCval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCrow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dCCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dCrow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dCcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dCval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dCrow, nnz, A_rows, dCCsrRowPtr);

  cusparseSpMatDescr_t matC;
  CHECK_CUSPARSE(cusparseCreateCsr(&matC,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dCCsrRowPtr,
                                   dCcol,
                                   dCval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_16F));

  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  void* dBuffer1 = NULL;
  void* dBuffer2 = NULL;
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

  // ask bufferSize1 bytes for external memory
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               CUDA_R_16F,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc,
                                               &bufferSize1,
                                               NULL))
  CHECK_CUDA(cudaMalloc((void**)&dBuffer1, bufferSize1))
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               CUDA_R_16F,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc,
                                               &bufferSize1,
                                               dBuffer1))

  // ask bufferSize2 bytes for external memory
  CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        matB,
                                        &beta,
                                        matC,
                                        CUDA_R_16F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize2,
                                        NULL))
  CHECK_CUDA(cudaMalloc((void**)&dBuffer2, bufferSize2))

  // compute the intermediate product of A * B
  CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        matB,
                                        &beta,
                                        matC,
                                        CUDA_R_16F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize2,
                                        dBuffer2))
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  CHECK_CUSPARSE(
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1))
  nnz_c = C_nnz1;
  // allocate matrix C
  int *dC_csrOffsets, *dC_columns;
  half* dC_values;
  CHECK_CUDA(cudaMalloc((void**)&dC_columns, C_nnz1 * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void**)&dC_values, C_nnz1 * sizeof(half)))
  CHECK_CUDA(cudaMalloc((void**)&dC_csrOffsets, (A_rows + 1) * sizeof(int)))
  // NOTE: if 'beta' != 0, the values of C must be update after the allocation
  //       of dC_values, and before the call of cusparseSpGEMM_copy

  // update matC with the new pointers
  CHECK_CUSPARSE(
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values))

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C
  CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha,
                                     matA,
                                     matB,
                                     &beta,
                                     matC,
                                     CUDA_R_16F,
                                     CUSPARSE_SPGEMM_DEFAULT,
                                     spgemmDesc))
  // device result check
  matC_roc = (half*)alpha_malloc(C_nnz1 * sizeof(half));
  // std::cout << "C_nnz1: " << C_nnz1 << std::endl;
  CHECK_CUDA(cudaMemcpy(
    matC_roc, dC_values, C_nnz1 * sizeof(half), cudaMemcpyDeviceToHost))
  // int* cst_ptr = (int*)alpha_malloc((A_rows + 1) * sizeof(int));
  // CHECK_CUDA(cudaMemcpy(
  //   cst_ptr, dC_csrOffsets, (A_rows + 1) * sizeof(int),
  //   cudaMemcpyDeviceToHost))
  // for(int i=0;i<A_rows + 1;i++){
  //   printf("+%d+, ",cst_ptr[i]);
  // }
  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroySpMat(matB))
  CHECK_CUSPARSE(cusparseDestroySpMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dBrow);
  cudaFree(dBcol);
  cudaFree(dBval);
  cudaFree(dCrow);
  cudaFree(dCcol);
  cudaFree(dCval);
}

static void
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  int nnz = rnnz;

  // Offload data to device
  int* dACsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  half* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dACsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dACsrRowPtr);

  alphasparseSpMatDescr_t matA;
  alphasparseCreateCsr(&matA,
                       A_rows,
                       A_cols,
                       nnz,
                       dACsrRowPtr,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_16F);

  int* dBCsrRowPtr = NULL;
  int* dBrow = NULL;
  int* dBcol = NULL;
  half* dBval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBrow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dBCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dBrow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dBcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dBval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dBrow, nnz, A_rows, dBCsrRowPtr);

  alphasparseSpMatDescr_t matB;
  alphasparseCreateCsr(&matB,
                       A_rows,
                       A_cols,
                       nnz,
                       dBCsrRowPtr,
                       dBcol,
                       dBval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_16F);

  int* dCCsrRowPtr = NULL;
  int* dCrow = NULL;
  int* dCcol = NULL;
  half* dCval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCrow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCval, sizeof(half) * nnz));
  PRINT_IF_CUDA_ERROR(
    cudaMalloc((void**)&dCCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(cudaMemcpy(
    dCrow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dCcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dCval, coo_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dCrow, nnz, A_rows, dCCsrRowPtr);

  alphasparseSpMatDescr_t matC;
  alphasparseCreateCsr(&matC,
                       A_rows,
                       A_cols,
                       nnz,
                       dCCsrRowPtr,
                       dCcol,
                       dCval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_16F);

  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  void* dBuffer1 = NULL;
  void* dBuffer2 = NULL;
  // SpGEMM Computation
  alphasparseSpGEMMDescr_t spgemmDesc;
  alphasparseSpGEMM_createDescr(&spgemmDesc);

  alphasparseSpGEMMreuse_compute(handle,
                                 ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                 ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 matA,
                                 matB,
                                 &beta,
                                 matC,
                                 ALPHA_R_32F,
                                 ALPHASPARSE_SPGEMM_DEFAULT,
                                 spgemmDesc);
  // allocate matrix C
  nnz_c = matC->nnz;
  matC_ict = (half*)alpha_malloc(matC->nnz * sizeof(half));
  CHECK_CUDA(cudaMemcpy(matC_ict,
                        matC->val_data,
                        matC->nnz * sizeof(half),
                        cudaMemcpyDeviceToHost))
  // int* matC_ptr = (int*)alpha_malloc(matC->nnz * sizeof(int));
  // CHECK_CUDA(cudaMemcpy(
  //   matC_ptr, matC->row_data, matC->rows * sizeof(int),
  //   cudaMemcpyDeviceToHost))
  // for (int i = 0; i < matC->rows; i++) {
  //   std::cout << matC_ptr[i] << ", ";
  // }
  // std::cout << std::endl;
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dBrow);
  cudaFree(dBcol);
  cudaFree(dBval);
  cudaFree(dCrow);
  cudaFree(dCcol);
  cudaFree(dCval);
}

int
main(int argc, const char* argv[])
{
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);

  // read coo
  alpha_read_coo<half>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, half>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, A_rows); // 默认C是方阵
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

  cuda_mm();
  alpha_mm();
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << matC_ict[i] << ", ";
  }
  check((half*)matC_roc, nnz_c, (half*)matC_ict, nnz_c);
  return 0;
}
