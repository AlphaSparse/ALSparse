
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
struct alpha_matrix_descr descrT;

int m, k, n, nnz;
long long ldb, ldc;
long long B_size, C_size;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
float* coo_values;

long long C_rows, C_cols;
long long B_rows, B_cols;
// coo format
float* B_val;
float* ict_C;
float* cuda_C;

// parms for kernel
const float alpha = 2.3f;

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
  bsrsm2Info_t info = 0;
  int pBufferSize;
  void *pBuffer = 0;
  const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
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

  cusparseFillMode_t FillMode;
  cusparseDiagType_t DiagType;
  int structural_zero;
  int numerical_zero;
  
  if(descrT.mode == ALPHA_SPARSE_FILL_MODE_LOWER) FillMode = CUSPARSE_FILL_MODE_LOWER;
  else FillMode = CUSPARSE_FILL_MODE_UPPER;
  if(descrT.diag == ALPHA_SPARSE_DIAG_NON_UNIT) DiagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
  else DiagType = CUSPARSE_DIAG_TYPE_UNIT;

  CHECK_CUSPARSE(cusparseSetMatFillMode(descrC, FillMode));
  CHECK_CUSPARSE(cusparseSetMatDiagType(descrC, DiagType));
  CHECK_CUSPARSE(cusparseCreateBsrsm2Info(&info));  

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValC, sizeof(float)*(blockDim*blockDim)*nnzb));
  CHECK_CUSPARSE(cusparseScsr2bsr(handle, dir, m, n,
                                  descrA, dAval, dCsrRowPtr, dAcol, blockDim,
                                  descrC, bsrValC, bsrRowPtrC, bsrColIndC));

  cusparseSbsrsm2_bufferSize(handle, dir, transA, transB, mb, n, nnzb, descrC,
              bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info, &pBufferSize);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&pBuffer, pBufferSize));   
  
  CHECK_CUSPARSE(cusparseSbsrsm2_analysis(handle, dir, transA, transB, mb, n, nnzb, descrC,
                                    bsrValC, bsrRowPtrC, bsrColIndC, blockDim,
                                    info, policy, pBuffer));   

  CHECK_CUDA(cudaMalloc((void**)&dB, B_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dC, C_size * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dB, B_val, B_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, cuda_C, C_size * sizeof(float), cudaMemcpyHostToDevice));  


  cusparseStatus_t status = cusparseXbsrsm2_zeroPivot(handle, info, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
  }
  printf("CUDA trans B %d\n", transB);
  CHECK_CUSPARSE(cusparseSbsrsm2_solve(handle, dir, transA, transB, mb, n, nnzb, &alpha,
              descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info, dB, ldb, dC, ldc, policy, pBuffer));

  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));
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
  alpha_bsrsv2Info_t info = ALPHA_SPARSE_OPAQUE;
  int pBufferSize;
  void *pBuffer = 0;
  const alphasparseSolvePolicy_t policy = ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL;

  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
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

  CHECK_CUDA(cudaMalloc((void**)&dB, B_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dC, C_size * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dB, B_val, B_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, ict_C, C_size * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseMatDescr_t descr_alpha ;
  alphasparseCreateMatDescr(&descr_alpha);
  alphasparseSetMatFillMode(descr_alpha, descrT.mode);
  alphasparseSetMatDiagType(descr_alpha, descrT.diag);
  printf("ALPHA trans B %d\n", transBT);
  alphasparseSbsrsm2_bufferSize(handle, dir_alpha, transAT, transBT, mb, n, nnzb, descr_alpha,
              bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info, &pBufferSize);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&pBuffer, pBufferSize)); 
  cudaMemset(pBuffer, 0, pBufferSize);

  alphasparseSbsrsm2_analysis(handle, dir_alpha, transAT, transBT, mb, n, nnzb, descr_alpha,
                                    bsrValC, bsrRowPtrC, bsrColIndC, blockDim,
                                    info, policy, &pBufferSize);   

  alphasparseSbsrsm2_solve(handle, dir_alpha, transAT, transBT, mb, n, nnzb, &alpha,
              descr_alpha, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info, dB, ldb, dC, ldc, policy, pBuffer);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(ict_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));
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
  descrT = alpha_args_get_matrix_descrA(argc, argv);
  dir_alpha = (alphasparseDirection_t)alpha_args_get_layout(argc, argv);
  // read coo
  alpha_read_coo<float>(
    file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);

  columns = args_get_cols(argc, argv, n);
  coo_order<int32_t, float>(nnz, coo_row_index, coo_col_index, coo_values);
  csrRowPtr = (int*)alpha_malloc(sizeof(int) * (m + 1));
  
  if (transAT == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
        if (transBT == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = m;
            C_cols = columns;
            B_cols = columns;
            ldb    = n;
            ldc    = m;
        } else {
            C_rows = columns;
            C_cols = m;
            B_cols = n;
            ldb    = columns;
            ldc    = columns;
        }
  } 
  else // transA, conjA
  {
    return 0;
  }
  
  B_size = ldb * B_cols;
  C_size = ldc * C_cols;
  B_val = (float*)alpha_malloc(B_size * sizeof(float));
  ict_C = (float*)alpha_malloc(C_size * sizeof(float));
  cuda_C = (float*)alpha_malloc(C_size * sizeof(float));

  alpha_fill_random(B_val, 0, B_size);
  alpha_fill_random(ict_C, 1, C_size);
  alpha_fill_random(cuda_C, 1, C_size);

  cuda_mv();
  alpha_mv();
  check((float*)cuda_C, C_size, (float*)ict_C, C_size);
 
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
