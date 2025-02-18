
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
#include "../../format/alphasparseXcsr2bsr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
int thread_num;
bool check_flag;
int iter;
long long columns;

alphasparseOperation_t transAT, transBT;
alphasparseDirection_t dir_alpha;

int m, k, n, nnz;
long long ldb, ldc;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
float* coo_values;

// coo format
float* B_val;
float* ict_C;
float* cuda_C;

// parms for kernel
const float alpha = 2.3f;
const float beta = 3.4f;

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
  float* dB = NULL;
  float* dC = NULL;
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  float* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(float) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));
  
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);

  int blockDim = 2;
  cusparseDirection_t dir;
  if(dir_alpha == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) dir = CUSPARSE_DIRECTION_ROW;
  else dir = CUSPARSE_DIRECTION_COLUMN;
  float* bsrValC = NULL;
  int* bsrRowPtrC = NULL;
  int* bsrColIndC = NULL;
  int nnzb; //base
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
  int kb = (k + blockDim-1)/blockDim;
  cusparseOperation_t transA, transB;
  if(transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transA = CUSPARSE_OPERATION_TRANSPOSE;
  else if(transAT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) transA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  if(transBT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transB = CUSPARSE_OPERATION_TRANSPOSE;
  else if(transBT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) transB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1)));
  // nnzTotalDevHostPtr points to host memory
  // int *nnzTotalDevHostPtr = &nnzb;

  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

  CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dir, m, n,
                                    descrA, dCsrRowPtr, dAcol, blockDim,
                                    descrC, bsrRowPtrC, &nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValC, sizeof(float)*(blockDim*blockDim)*nnzb));
  CHECK_CUSPARSE(cusparseScsr2bsr(handle, dir, m, n,
                                  descrA, dAval, dCsrRowPtr, dAcol, blockDim,
                                  descrC, bsrValC, bsrRowPtrC, bsrColIndC));

  CHECK_CUDA(cudaMalloc((void**)&dB, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dC, m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dB, B_val, k * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, cuda_C, m * n * sizeof(float), cudaMemcpyHostToDevice));  

  CHECK_CUSPARSE(cusparseSbsrmm(handle, dir, transA, transB, mb, n, kb, nnzb, &alpha,
              descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, dB, ldb, &beta, dC, ldc));
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
  // Clear up on device
  cusparseDestroyMatDescr(descrA);  
  cusparseDestroyMatDescr(descrC);  
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(bsrValC);
  cudaFree(bsrRowPtrC);
  cudaFree(bsrColIndC);
  cusparseDestroy(handle);
}

static void
alpha_mv()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);
  cusparseHandle_t chandle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&chandle));

  // Offload data to device
  float* dB = NULL;
  float* dC = NULL;
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  float* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(float) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));

  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);

  int blockDim = 2;
  cusparseDirection_t dir;
  if(dir_alpha == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) dir = CUSPARSE_DIRECTION_ROW;
  else dir = CUSPARSE_DIRECTION_COLUMN;
  float* bsrValC = NULL;
  int* bsrRowPtrC = NULL;
  int* bsrColIndC = NULL;
  int nnzb; //base
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
  int kb = (k + blockDim-1)/blockDim;
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1)));
  // nnzTotalDevHostPtr points to host memory
  // int *nnzTotalDevHostPtr = &nnzb;

  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

  CHECK_CUSPARSE(cusparseXcsr2bsrNnz(chandle, dir, m, n,
                                    descrA, dCsrRowPtr, dAcol, blockDim,
                                    descrC, bsrRowPtrC, &nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValC, sizeof(float)*(blockDim*blockDim)*nnzb));
  cusparseScsr2bsr(chandle, dir, m, n,
                    descrA, dAval, dCsrRowPtr, dAcol, blockDim,
                    descrC, bsrValC, bsrRowPtrC, bsrColIndC);

  CHECK_CUDA(cudaMalloc((void**)&dB, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dC, m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dB, B_val, k * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, ict_C, m * n * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseMatDescr_t descr_alpha ;
  alphasparseCreateMatDescr(&descr_alpha);

  alphasparseSbsrmm(handle, dir_alpha, transAT, transBT, mb, n, kb, nnzb, &alpha,
              descr_alpha, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, dB, ldb, &beta, dC, ldc);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(ict_C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
  // Clear up on device
  cusparseDestroyMatDescr(descrA);  
  cusparseDestroyMatDescr(descrC);  
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(bsrValC);
  cudaFree(bsrRowPtrC);
  cudaFree(bsrColIndC);
  cusparseDestroy(chandle);
  cudaDeviceSynchronize();
}

int
main(int argc, const char* argv[])
{
  // args
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transAT = alpha_args_get_transA(argc, argv);
  transBT = alpha_args_get_transB(argc, argv);
  dir_alpha = (alphasparseDirection_t)alpha_args_get_layout(argc, argv);
  // read coo
  alpha_read_coo<float>(
    file, &m, &k, &nnz, &coo_row_index, &coo_col_index, &coo_values);

  n = args_get_cols(argc, argv, k);
  coo_order<int32_t, float>(nnz, coo_row_index, coo_col_index, coo_values);
  csrRowPtr = (int*)alpha_malloc(sizeof(int) * (m + 1));
  if (transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transAT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    int temp = n;
    n = m;
    m = temp;
  }
  ldb = k;
  ldc = m;
  B_val = (float*)alpha_malloc(k * n * sizeof(float));
  ict_C = (float*)alpha_malloc(m * n * sizeof(float));
  cuda_C = (float*)alpha_malloc(m * n * sizeof(float));

  alpha_fill_random(B_val, 0, k * n);
  alpha_fill_random(ict_C, 1, m * n);
  alpha_fill_random(cuda_C, 1, m * n);
  cuda_mv();
  alpha_mv();
  check((float*)cuda_C, m * n, (float*)ict_C, m * n);
  for (int i = 0; i < 20; i++) {
    std::cout << cuda_C[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << ict_C[i] << ", ";
  }
  std::cout << std::endl;
  return 0;
}
