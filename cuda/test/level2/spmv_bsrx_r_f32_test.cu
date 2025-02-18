
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

alphasparseOperation_t transA;
alphasparseDirection_t dir_alpha;

int m, n, nnz;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
float* coo_values;

// coo format
float* x_val;
float* ict_y;
float* cuda_y;
int * h_bsrMaskPtr;
int sizeOfMask;

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
  float* dX = NULL;
  float* dY = NULL;
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
  int* bsrEndPtrC = NULL;
  int* h_bsrRowPtrC = NULL;
  int* bsrColIndC = NULL;
  int nnzb; //base
  int * bsrMaskPtr = NULL;
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
  int nb = (n + blockDim-1)/blockDim;
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1)));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrEndPtrC, sizeof(int) *(mb+1)));
  h_bsrRowPtrC = (int *)malloc(sizeof(int) *(mb+1));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrMaskPtr, sizeof(int) * sizeOfMask));
  // nnzTotalDevHostPtr points to host memory
  // int *nnzTotalDevHostPtr = &nnzb;
  CHECK_CUDA(cudaMemcpy(bsrMaskPtr, h_bsrMaskPtr, sizeof(int) *sizeOfMask, cudaMemcpyHostToDevice));

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

  CHECK_CUDA(cudaMemcpy(h_bsrRowPtrC, bsrRowPtrC, sizeof(int) *(mb+1), cudaMemcpyDeviceToHost));

  for(int i = 0; i < mb + 1; i++)
  {
    h_bsrRowPtrC[i] += 1;
  }

  CHECK_CUDA(cudaMemcpy(bsrEndPtrC, h_bsrRowPtrC, sizeof(int) *(mb+1), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(float), cudaMemcpyHostToDevice));  

  CHECK_CUSPARSE(cusparseSbsrxmv(handle, dir, transA, sizeOfMask, mb, nb, nnzb, &alpha,
              descrC, bsrValC, bsrMaskPtr, bsrRowPtrC, bsrEndPtrC, bsrColIndC, blockDim, dX, &beta, dY));
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
  // Clear up on device
  cusparseDestroyMatDescr(descrA);  
  cusparseDestroyMatDescr(descrC);  
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dX);
  cudaFree(dY);
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
  float* dX = NULL;
  float* dY = NULL;
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
  int* bsrEndPtrC = NULL;
  int* h_bsrRowPtrC = NULL;
  int* bsrColIndC = NULL;
  int nnzb; //base  
  int * bsrMaskPtr = NULL;
  
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
  int nb = (n + blockDim-1)/blockDim;
  alphasparseOperation_t transA = ALPHA_SPARSE_OPERATION_NON_TRANSPOSE;
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1)));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrEndPtrC, sizeof(int) *(mb+1)));
  h_bsrRowPtrC = (int *)malloc(sizeof(int) *(mb+1));
  
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrMaskPtr, sizeof(int) * sizeOfMask));
  // nnzTotalDevHostPtr points to host memory
  // int *nnzTotalDevHostPtr = &nnzb;  

  CHECK_CUDA(cudaMemcpy(bsrMaskPtr, h_bsrMaskPtr, sizeof(int) *sizeOfMask, cudaMemcpyHostToDevice));

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

  CHECK_CUDA(cudaMemcpy(h_bsrRowPtrC, bsrRowPtrC, sizeof(int) *(mb+1), cudaMemcpyDeviceToHost));

  for(int i = 0; i < mb + 1; i++)
  {
    h_bsrRowPtrC[i] += 1;
  }

  CHECK_CUDA(cudaMemcpy(bsrEndPtrC, h_bsrRowPtrC, sizeof(int) *(mb+1), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseMatDescr_t descr_alpha ;
  alphasparseCreateMatDescr(&descr_alpha);

  alphasparseSbsrxmv(handle, dir_alpha, transA, sizeOfMask, mb, nb, nnzb, &alpha,
              descr_alpha, bsrValC, bsrMaskPtr, bsrRowPtrC, bsrEndPtrC, bsrColIndC, blockDim, dX, &beta, dY);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(ict_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
  // Clear up on device
  cusparseDestroyMatDescr(descrA);  
  cusparseDestroyMatDescr(descrC);  
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dX);
  cudaFree(dY);
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
  transA = alpha_args_get_transA(argc, argv);
  dir_alpha = (alphasparseDirection_t)alpha_args_get_layout(argc, argv);

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
  x_val = (float*)alpha_malloc(n * sizeof(float));
  ict_y = (float*)alpha_malloc(m * sizeof(float));
  cuda_y = (float*)alpha_malloc(m * sizeof(float));

  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  sizeOfMask = (m / 2) / 4;
  h_bsrMaskPtr = (int *)malloc(sizeof(int) * sizeOfMask);
  for(int i = 0; i < sizeOfMask; i++)
  {
    if(i == 0) h_bsrMaskPtr[i] = rand() % 4;
    else h_bsrMaskPtr[i] = h_bsrMaskPtr[i - 1] + rand() % 4 + 1;
  }
  for(int i = 0; i < 5; i++)
    std::cout<<"mask "<<h_bsrMaskPtr[i]<<std::endl;

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
  std::cout << std::endl;
  return 0;
}
