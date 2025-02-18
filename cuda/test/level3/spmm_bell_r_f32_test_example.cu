
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
float *coo_values;

// parms for kernel
float *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
long long B_rows, B_cols;
const float alpha = 2.f;
const float beta = 3.f;
int blocksize;

int   A_num_rows      = 4;
int   A_num_cols      = 4;
int   A_ell_blocksize = 2;
int   A_ell_cols      = 2;
int   A_num_blocks    = A_ell_cols * A_num_rows /
                      (A_ell_blocksize * A_ell_blocksize);
int   B_num_rows      = A_num_cols;
int   B_num_cols      = 3;
int   ldb             = B_num_rows;
int   ldc             = A_num_rows;
int   B_size          = ldb * B_num_cols;
int   C_size          = ldc * B_num_cols;
int   hA_columns[]    = { 1, 0};
float hA_values[]    = { 1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f};
float hB[]           = { 1.0f,  2.0f,  3.0f,  4.0f,
                        5.0f,  6.0f,  7.0f,  8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f };
float hC[]           = { 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f };
float hC_result[]    = { 11.0f, 25.0f,  17.0f,  23.0f,
                        23.0f, 53.0f,  61.0f,  83.0f,
                        35.0f, 81.0f, 105.0f, 143.0f };

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
  int    *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,
                                  A_ell_cols * A_num_rows * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

  CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                          A_num_blocks * sizeof(int),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                          A_ell_cols * A_num_rows * sizeof(float),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                          cudaMemcpyHostToDevice) )

  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  // Create sparse matrix A in blocked ELL format
  CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                    &matA,
                                    A_num_rows, A_num_cols, A_ell_blocksize,
                                    A_ell_cols, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  // Create dense matrix B
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) )
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) )
  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  // execute SpMM
  CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA( cudaMemcpy(matC_roc, dC, C_size * sizeof(float),
                          cudaMemcpyDeviceToHost) )
  // Clear up on device
  CHECK_CUDA( cudaFree(dBuffer) )
  CHECK_CUDA( cudaFree(dA_columns) )
  CHECK_CUDA( cudaFree(dA_values) )
  CHECK_CUDA( cudaFree(dB) )
  CHECK_CUDA( cudaFree(dC) )
}

static void
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  int *dArow = NULL;
  int *dAcol = NULL;
  float *dAval = NULL;

  int nnz = rnnz;

  float *dmatB = NULL;
  float *dmatC = NULL;

  cudaMalloc((void **)&dmatB, sizeof(float) * B_size);
  cudaMalloc((void **)&dmatC, sizeof(float) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAval, sizeof(float) * nnz));

  CHECK_CUDA(cudaMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseSpMatDescr_t coo;
  alphasparseCreateCoo(&coo,
                       A_num_rows,
                       A_num_cols,
                       nnz,
                       dArow,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_32F);

  alphasparseSpMatDescr_t bell;
  alphasparseCoo2bell<int, float>(coo, bell, A_ell_blocksize);

  int    *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,
                                  A_ell_cols * A_num_rows * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

  CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                          A_num_blocks * sizeof(int),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                          A_ell_cols * A_num_rows * sizeof(float),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                          cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                          cudaMemcpyHostToDevice) )


  alphasparseSpMatDescr_t matA;
  alphasparseDnMatDescr_t matB, matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  // Create sparse matrix A in blocked ELL format
  alphasparseCreateBlockedEll(
                              &matA,
                              A_num_rows, A_num_cols, A_ell_blocksize,
                              A_ell_cols, dA_columns, dA_values,
                              ALPHA_SPARSE_INDEXTYPE_I32,
                              ALPHA_SPARSE_INDEX_BASE_ZERO, ALPHA_R_32F);
  // Create dense matrix B
  alphasparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                      ALPHA_R_32F, ALPHASPARSE_ORDER_COL);
  // Create dense matrix C
  alphasparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                      ALPHA_R_32F, ALPHASPARSE_ORDER_COL);
  // allocate an external buffer if needed
  alphasparseSpMM_bufferSize(
                                handle,
                                ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, ALPHA_R_32F,
                                ALPHASPARSE_SPMM_ALG_DEFAULT, &bufferSize);
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  // execute SpMM
  alphasparseSpMM(handle,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha, matA, matB, &beta, matC, ALPHA_R_32F,
                  ALPHASPARSE_SPMM_ALG_DEFAULT, dBuffer) ;

  //--------------------------------------------------------------------------
  // device result check
  CHECK_CUDA( cudaMemcpy(matC_ict, dC, C_size * sizeof(float),
                          cudaMemcpyDeviceToHost) )
  // Clear up on device
  CHECK_CUDA( cudaFree(dBuffer) )
  CHECK_CUDA( cudaFree(dA_columns) )
  CHECK_CUDA( cudaFree(dA_values) )
  CHECK_CUDA( cudaFree(dB) )
  CHECK_CUDA( cudaFree(dC) )
}

int main(int argc, const char *argv[])
{
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  alpha_read_coo<float>(
      file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, float>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = B_num_cols;
 
  matC_ict = (float *)alpha_malloc(C_size * sizeof(float));
  matC_roc = (float *)alpha_malloc(C_size * sizeof(float));
  // init x y
  // init B C
  cuda_mm();
  alpha_mm();

  for (int i = 0; i < C_size; i++)
  {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < C_size; i++)
  {
    std::cout << matC_ict[i] << ", ";
  }
  std::cout << std::endl;
  check((float *)matC_roc, C_size, (float *)matC_ict, C_size);
  return 0;
}
