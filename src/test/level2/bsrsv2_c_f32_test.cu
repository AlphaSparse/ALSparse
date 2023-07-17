
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

alphasparseOperation_t transAT;
alphasparseDirection_t dir_alpha;
struct alpha_matrix_descr descrT;

int m, n, nnz;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
cuFloatComplex* coo_values;

// coo format
cuFloatComplex* x_val;
cuFloatComplex* ict_y;
cuFloatComplex* cuda_y;

// parms for kernel
const cuFloatComplex alpha = {2.3f, 3.4f};

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
cuda_sv2()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  cuFloatComplex* dX = NULL;
  cuFloatComplex* dY = NULL;
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  cuFloatComplex* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(cuFloatComplex) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));
  
  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);

  int blockDim = 2;
  cusparseDirection_t dir;
  if(dir_alpha == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) dir = CUSPARSE_DIRECTION_ROW;
  else dir = CUSPARSE_DIRECTION_COLUMN;
  cuFloatComplex* bsrValC = NULL;
  int* bsrRowPtrC = NULL;
  int* bsrColIndC = NULL;
  int nnzb; //base
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
  // int nb = (n + blockDim-1)/blockDim;
  cusparseOperation_t transA;
  if(transAT == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if(transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transA = CUSPARSE_OPERATION_TRANSPOSE;
  else transA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1)));
  // nnzTotalDevHostPtr points to host memory
  // int *nnzTotalDevHostPtr = &nnzb;

  bsrsv2Info_t info = 0;
  int pBufferSize;
  void *pBuffer = 0;

  int structural_zero;
  int numerical_zero;
  const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));  
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

  CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dir, m, n,
                                    descrA, dCsrRowPtr, dAcol, blockDim,
                                    descrC, bsrRowPtrC, &nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb));

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValC, sizeof(cuFloatComplex)*(blockDim*blockDim)*nnzb));
  CHECK_CUSPARSE(cusparseCcsr2bsr(handle, dir, m, n,
                                  descrA, dAval, dCsrRowPtr, dAcol, blockDim,
                                  descrC, bsrValC, bsrRowPtrC, bsrColIndC));
  cusparseFillMode_t FillMode;
  cusparseDiagType_t DiagType;
  
  if(descrT.mode == ALPHA_SPARSE_FILL_MODE_LOWER) FillMode = CUSPARSE_FILL_MODE_LOWER;
  else FillMode = CUSPARSE_FILL_MODE_UPPER;
  if(descrT.diag == ALPHA_SPARSE_DIAG_NON_UNIT) DiagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
  else DiagType = CUSPARSE_DIAG_TYPE_UNIT;

  CHECK_CUSPARSE(cusparseSetMatFillMode(descrC, FillMode));
  CHECK_CUSPARSE(cusparseSetMatDiagType(descrC, DiagType));
  CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info));

  cusparseCbsrsv2_bufferSize(handle, dir, transA, mb, nnzb, descrC,
              bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info, &pBufferSize);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&pBuffer, pBufferSize));   
  
  CHECK_CUSPARSE(cusparseCbsrsv2_analysis(handle, dir, transA, mb, nnzb, descrC,
                                    bsrValC, bsrRowPtrC, bsrColIndC, blockDim,
                                    info, policy, pBuffer));   

  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));    

  // L has unit diagonal, so no structural zero is reported.
  cusparseStatus_t status = cusparseXbsrsv2_zeroPivot(handle, info, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
  }
  // step 5: solve L*y = x
  CHECK_CUSPARSE(cusparseCbsrsv2_solve(handle, dir, transA, mb, nnzb, &alpha, descrC,
                                  bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info,
                                  dX, dY, policy, pBuffer));
  // L has unit diagonal, so no numerical zero is reported.
  status = cusparseXbsrsv2_zeroPivot(handle, info, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }     

  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(cuFloatComplex) * m, cudaMemcpyDeviceToHost));
  // Clear up on device
  cudaFree(pBuffer);
  cusparseDestroyBsrsv2Info(info);
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
alpha_sv2()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);
  cusparseHandle_t chandle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&chandle));

  // Offload data to device
  cuFloatComplex* dX = NULL;
  cuFloatComplex* dY = NULL;
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  cuFloatComplex* dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(cuFloatComplex) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));

  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);

  int blockDim = 2;
  cusparseDirection_t dir;
  if(dir_alpha == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) dir = CUSPARSE_DIRECTION_ROW;
  else dir = CUSPARSE_DIRECTION_COLUMN;
  cuFloatComplex* bsrValC = NULL;
  int* bsrRowPtrC = NULL;
  int* bsrColIndC = NULL;
  int nnzb; //base
  cusparseMatDescr_t descrA;
  cusparseMatDescr_t descrC;
  int mb = (m + blockDim-1)/blockDim;
  // int nb = (n + blockDim-1)/blockDim;
  alpha_bsrsv2Info_t info = ALPHA_SPARSE_OPAQUE;
  int pBufferSize = m;
  void *pBuffer = 0;
  const alphasparseSolvePolicy_t policy = ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL;

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

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValC, sizeof(cuFloatComplex)*(blockDim*blockDim)*nnzb));
  cusparseCcsr2bsr(chandle, dir, m, n,
                    descrA, dAval, dCsrRowPtr, dAcol, blockDim,
                    descrC, bsrValC, bsrRowPtrC, bsrColIndC);

  CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

  alphasparseMatDescr_t descr_alpha ;
  alphasparseCreateMatDescr(&descr_alpha);
  alphasparseSetMatFillMode(descr_alpha, descrT.mode);
  alphasparseSetMatDiagType(descr_alpha, descrT.diag);

  if(descrT.diag == ALPHA_SPARSE_DIAG_UNIT) pBufferSize *= 2;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&pBuffer, sizeof(cuFloatComplex) * pBufferSize)); 
  cudaMemset(pBuffer, 0, sizeof(cuFloatComplex) * pBufferSize);

  alphasparseCbsrsv2_solve(handle, dir_alpha, transAT, mb, nnzb, &alpha, descr_alpha,
                                  bsrValC, bsrRowPtrC, bsrColIndC, blockDim, info,
                                  dX, dY, policy, pBuffer);
  
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemcpy(ict_y, dY, sizeof(cuFloatComplex) * m, cudaMemcpyDeviceToHost));
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
  cudaFree(pBuffer);
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
  descrT = alpha_args_get_matrix_descrA(argc, argv);
  dir_alpha = (alphasparseDirection_t)alpha_args_get_layout(argc, argv);

  // read coo
  alpha_read_coo<cuFloatComplex>(
    file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, cuFloatComplex>(nnz, coo_row_index, coo_col_index, coo_values);
  csrRowPtr = (int*)alpha_malloc(sizeof(int) * (m + 1));
  if (transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transAT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
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
  x_val = (cuFloatComplex*)alpha_malloc(n * sizeof(cuFloatComplex));
  ict_y = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));
  cuda_y = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));

  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  cuda_sv2();
  alpha_sv2();
  check((cuFloatComplex*)cuda_y, m, (cuFloatComplex*)ict_y, m);
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
