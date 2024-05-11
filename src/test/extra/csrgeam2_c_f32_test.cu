#include "../test_common.h"

/**
 * @brief ict csr geam2 test
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

const char* fileA; 
const char* fileB;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transAT;
alphasparseOperation_t transBT;

long long Cnnz_ict, Cnnz_cuda;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
cuFloatComplex* coo_values;

int B_rows, B_cols, Bnnz;
int *B_coo_row_index, *B_coo_col_index;
cuFloatComplex* B_coo_values;

// parms for kernel
cuFloatComplex *csrValC_ict, *csrValC_cuda;
const cuFloatComplex alpha = {1.1f,2.4f};
const cuFloatComplex beta = {3.2f,4.3f};

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

static void cuda_csrgeam()
{
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    int baseC, nnzC;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    size_t BufferSizeInBytes;
    char *buffer = NULL;
    int *nnzTotalDevHostPtr = &nnzC;
    int m = A_rows, n = A_cols;

    int* csrRowPtrA = NULL;
    int* dArow = NULL;
    int* csrColIndA = NULL;
    cuFloatComplex* csrValA = NULL;

    int* csrRowPtrB = NULL;
    int* dBrow = NULL;
    int* csrColIndB = NULL;
    cuFloatComplex* csrValB = NULL;

    int* csrRowPtrC = NULL;
    int* csrColIndC = NULL;
    cuFloatComplex* csrValC = NULL;

    int nnzA = rnnz;
    int nnzB = Bnnz;

    cusparseMatDescr_t descrA;
    cusparseMatDescr_t descrB;
    cusparseMatDescr_t descrC;

    cusparseOperation_t transA, transB;
    if(transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transA = CUSPARSE_OPERATION_TRANSPOSE;
    else if(transAT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) transA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    else transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    if(transBT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transB = CUSPARSE_OPERATION_TRANSPOSE;
    else if(transBT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) transB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    else transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnzA));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&csrColIndA, sizeof(int) * nnzA));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&csrValA, sizeof(cuFloatComplex) * nnzA));

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dBrow, sizeof(int) * nnzB));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&csrColIndB, sizeof(int) * nnzB));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&csrValB, sizeof(cuFloatComplex) * nnzB));

    PRINT_IF_CUDA_ERROR(
        cudaMalloc((void**)&csrRowPtrA, sizeof(int) * (m + 1)));
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnzA * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        csrColIndA, coo_col_index, nnzA * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        csrValA, coo_values, nnzA * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    cusparseXcoo2csr(handle, dArow, nnzA, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);

    PRINT_IF_CUDA_ERROR(
        cudaMalloc((void**)&csrRowPtrB, sizeof(int) * (m + 1)));
    CHECK_CUDA(cudaMemcpy(
        dBrow, B_coo_row_index, nnzB * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        csrColIndB, B_coo_col_index, nnzB * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(csrValB, B_coo_values, nnzB * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    cusparseXcoo2csr(handle, dBrow, nnzB, m, csrRowPtrB, CUSPARSE_INDEX_BASE_ZERO);

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
    /* prepare buffer */
    CHECK_CUSPARSE(cusparseCcsrgeam2_bufferSizeExt(handle, m, n,
                                    &alpha,
                                    descrA, nnzA,
                                    csrValA, csrRowPtrA, csrColIndA,
                                    &beta,
                                    descrB, nnzB,
                                    csrValB, csrRowPtrB, csrColIndB,
                                    descrC,
                                    csrValC, csrRowPtrC, csrColIndC,
                                    &BufferSizeInBytes));

    CHECK_CUDA(cudaMalloc((void**)&buffer, sizeof(char)*BufferSizeInBytes));
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, m, n,
                        descrA, nnzA, csrRowPtrA, csrColIndA,
                        descrB, nnzB, csrRowPtrB, csrColIndB,
                        descrC, csrRowPtrC, nnzTotalDevHostPtr,
                        buffer));
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }
    Cnnz_cuda = nnzC;
    csrValC_cuda = (cuFloatComplex *)malloc(sizeof(cuFloatComplex)*nnzC);

    cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
    cudaMalloc((void**)&csrValC, sizeof(cuFloatComplex)*nnzC);

    CHECK_CUSPARSE(cusparseCcsrgeam2(handle, m, n,
                    &alpha,
                    descrA, nnzA,
                    csrValA, csrRowPtrA, csrColIndA,
                    &beta,
                    descrB, nnzB,
                    csrValB, csrRowPtrB, csrColIndB,
                    descrC,
                    csrValC, csrRowPtrC, csrColIndC,
                    buffer));

    CHECK_CUDA(cudaMemcpy(csrValC_cuda, csrValC, sizeof(cuFloatComplex)*nnzC, cudaMemcpyDeviceToHost));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
}

static void alpha_csrgeam()
{
    alphasparseHandle_t handle = NULL;
    initHandle(&handle);
    alphasparseGetHandle(&handle);
    int baseC, nnzC;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    size_t BufferSizeInBytes;
    char *buffer = NULL;
    int *nnzTotalDevHostPtr = &nnzC;
    int m = A_rows, n = A_cols;

    int* csrRowPtrA = NULL;
    int* dArow = NULL;
    int* csrColIndA = NULL;
    cuFloatComplex* csrValA = NULL;

    int* csrRowPtrB = NULL;
    int* dBrow = NULL;
    int* csrColIndB = NULL;
    cuFloatComplex* csrValB = NULL;

    int* csrRowPtrC = NULL;
    int* csrColIndC = NULL;
    cuFloatComplex* csrValC = NULL;

    int nnzA = rnnz;
    int nnzB = Bnnz;

    alphasparseMatDescr_t descrA;
    alphasparseMatDescr_t descrB;
    alphasparseMatDescr_t descrC;

    alphasparseCreateMatDescr(&descrA);
    alphasparseCreateMatDescr(&descrB);
    alphasparseCreateMatDescr(&descrC);

    cudaMalloc((void**)&dArow, sizeof(int) * nnzA);
    cudaMalloc((void**)&csrColIndA, sizeof(int) * nnzA);
    cudaMalloc((void**)&csrValA, sizeof(cuFloatComplex) * nnzA);

    cudaMalloc((void**)&dBrow, sizeof(int) * nnzB);
    cudaMalloc((void**)&csrColIndB, sizeof(int) * nnzB);
    cudaMalloc((void**)&csrValB, sizeof(cuFloatComplex) * nnzB);
    
    cudaMalloc((void**)&csrRowPtrA, sizeof(int) * (m + 1));
    cudaMemcpy(
        dArow, coo_row_index, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(
        csrColIndA, coo_col_index, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(csrValA, coo_values, nnzA * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    alphasparseXcoo2csr(dArow, nnzA, m, csrRowPtrA);
    
    cudaMalloc((void**)&csrRowPtrB, sizeof(int) * (m + 1));
    cudaMemcpy(
        dBrow, B_coo_row_index, nnzB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(
        csrColIndB, B_coo_col_index, nnzB * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(csrValB, B_coo_values, nnzB * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    alphasparseXcoo2csr(dBrow, nnzB, m, csrRowPtrB);

    cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
    /* prepare buffer */
    alphasparseCcsrgeam2_bufferSizeExt(handle, m, n,
                                    &alpha,
                                    descrA, nnzA,
                                    csrValA, csrRowPtrA, csrColIndA,
                                    &beta,
                                    descrB, nnzB,
                                    csrValB, csrRowPtrB, csrColIndB,
                                    descrC,
                                    csrValC, csrRowPtrC, csrColIndC,
                                    &BufferSizeInBytes);

    cudaMalloc((void**)&buffer, sizeof(char)*BufferSizeInBytes);
    alphasparseXcsrgeam2Nnz(handle, m, n,
                        descrA, nnzA, csrRowPtrA, csrColIndA,
                        descrB, nnzB, csrRowPtrB, csrColIndB,
                        descrC, csrRowPtrC, nnzTotalDevHostPtr,
                        buffer);
                        
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }
    Cnnz_ict = nnzC;
    csrValC_ict = (cuFloatComplex *)malloc(sizeof(cuFloatComplex)*nnzC);

    cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
    cudaMalloc((void**)&csrValC, sizeof(cuFloatComplex)*nnzC);

    alphasparseCcsrgeam2(handle, m, n,
                    &alpha,
                    descrA, nnzA,
                    csrValA, csrRowPtrA, csrColIndA,
                    &beta,
                    descrB, nnzB,
                    csrValB, csrRowPtrB, csrColIndB,
                    descrC,
                    csrValC, csrRowPtrC, csrColIndC,
                    buffer);

    cudaMemcpy(csrValC_ict, csrValC, sizeof(cuFloatComplex)*nnzC, cudaMemcpyDeviceToHost);

    // alphasparseDestroyMatDescr(descrA);
    // alphasparseDestroyMatDescr(descrB);
    // alphasparseDestroyMatDescr(descrC);
}

int
main(int argc, const char* argv[])
{
  args_help(argc, argv);
  fileA = args_get_data_fileA(argc, argv);
  fileB = args_get_data_fileB(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transAT = alpha_args_get_transA(argc, argv);
  transBT = alpha_args_get_transB(argc, argv);

  // read coo
  alpha_read_coo<cuFloatComplex>(
    fileA, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  alpha_read_coo<cuFloatComplex>(
    fileB, &B_rows, &B_cols, &Bnnz, &B_coo_row_index, &B_coo_col_index, &B_coo_values);
  
  if(A_rows != B_rows || A_cols != B_cols)
  {
    printf("INVALID SIZE OF MATRIX A AND B!\n");
    return 0;
  }
  coo_order<int32_t, cuFloatComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
  coo_order<int32_t, cuFloatComplex>(Bnnz, B_coo_row_index, B_coo_col_index, B_coo_values);

  cuda_csrgeam();
  CHECK_CUDA(cudaDeviceSynchronize());
  alpha_csrgeam();

  std::cout.precision( 10 );
  for (int i = 0; i < 20; i++) {
    std::cout << csrValC_cuda[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << csrValC_ict[i] << ", ";
  }
  check((cuFloatComplex*)csrValC_cuda, Cnnz_cuda, (cuFloatComplex*)csrValC_ict, Cnnz_ict);

  return 0;
}

