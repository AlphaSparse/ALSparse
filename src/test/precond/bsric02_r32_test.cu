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
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseDirection_t dir_alpha;
struct alpha_matrix_descr descrT;

int m, n, nnz, blockdim = 2;
int cuda_nnzb, ict_nnzb;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
float* coo_values;

// coo format
float* x_val;
float* ict_y;
float* cuda_y;

float* ict_val;
float* cuda_val;

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

void cuda_ilu02()
{
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Offload data to device
    int* d_csrRowPtr = NULL;
    int* dArow = NULL;
    int* d_csrColInd = NULL;
    float* d_csrVal = NULL;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrVal, sizeof(float) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (m + 1)));
    
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_csrColInd, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_csrVal, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, d_csrRowPtr);
    cusparseDirection_t dir;
    if(dir_alpha == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) dir = CUSPARSE_DIRECTION_ROW;
    else dir = CUSPARSE_DIRECTION_COLUMN;
    float* bsrValA = NULL;
    int* bsrRowPtrA = NULL;
    int* bsrColIndA = NULL;
    int nnzb; //base
    cusparseMatDescr_t descr;
    cusparseMatDescr_t descrA;
    int mb = (m + blockdim-1)/blockdim;
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrA, sizeof(int) *(mb+1)));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));  
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dir, m, n,
                                      descr, d_csrRowPtr, d_csrColInd, blockdim,
                                      descrA, bsrRowPtrA, &nnzb));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrColIndA, sizeof(int)*nnzb));

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValA, sizeof(float)*(blockdim*blockdim)*nnzb));
    CHECK_CUSPARSE(cusparseScsr2bsr(handle, dir, m, n,
                                    descr, d_csrVal, d_csrRowPtr, d_csrColInd, blockdim,
                                    descrA, bsrValA, bsrRowPtrA, bsrColIndA));

    bsric02Info_t info_M = 0;

    int pBufferSize_M;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal

    // step 2: create a empty info structure
    // we need one info for bsric02 and two info's for csrsv2
    CHECK_CUSPARSE(cusparseCreateBsric02Info(&info_M));
    // step 3: query how much memory used in bsric02 and csrsv2, and allocate the buffer
    CHECK_CUSPARSE(cusparseSbsric02_bufferSize(handle, dir, mb, nnzb,
    descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockdim, info_M, &pBufferSize_M));
    pBufferSize = pBufferSize_M;
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);
    // step 4: perform analysis of incomplete Cholesky on M
    // perform analysis of triangular solve on L
    // perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of bsric02 and csrsv2 simultaneously.
    CHECK_CUSPARSE(cusparseSbsric02_analysis(handle, dir, mb, nnzb, descrA,
    bsrValA, bsrRowPtrA, bsrColIndA, blockdim, info_M,
    policy_M, pBuffer));
    cusparseStatus_t status = cusparseXbsric02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    // step 5: M = L * L'
    CHECK_CUSPARSE(cusparseSbsric02(handle, dir, mb, nnzb, descrA,
    bsrValA, bsrRowPtrA, bsrColIndA, blockdim, info_M, policy_M, pBuffer));
    status = cusparseXbsric02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }
    // step 6: solve L*z = x
    // CHECK_CUSPARSE(cusparseScsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
    // d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
    // d_x, d_z, policy_L, pBuffer));
    // // step 7: solve L'*y = z
    // CHECK_CUSPARSE(cusparseScsrsv2_solve(handle, trans_Lt, m, nnz, &alpha, descr_L,
    // d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt,
    // d_z, d_y, policy_Lt, pBuffer));
    free(cuda_val);
    cuda_val = (float *)malloc(sizeof(float)*nnzb);
    cudaMemcpy(cuda_val, bsrValA, sizeof(float)*nnzb, cudaMemcpyDeviceToHost);
    cuda_nnzb = nnzb;
    // step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyBsric02Info(info_M);
    cusparseDestroy(handle);
}

void alpha_ilu02()
{
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    cusparseHandle_t chandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&chandle));

    // Offload data to device
    int* d_csrRowPtr = NULL;
    int* dArow = NULL;
    int* d_csrColInd = NULL;
    float* d_csrVal = NULL;
 
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrVal, sizeof(float) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (m + 1)));

    cusparseDirection_t dir;
    if(dir_alpha == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) dir = CUSPARSE_DIRECTION_ROW;
    else dir = CUSPARSE_DIRECTION_COLUMN;
    
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_csrColInd, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_csrVal, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, d_csrRowPtr);
    alpha_bsric02Info_t info_M = ALPHA_SPARSE_OPAQUE;

    float* bsrValA = NULL;
    int* bsrRowPtrA = NULL;
    int* bsrColIndA = NULL;
    int nnzb; //base
    cusparseMatDescr_t descr;
    cusparseMatDescr_t descrA;
    int mb = (m + blockdim-1)/blockdim;
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrRowPtrA, sizeof(int) *(mb+1)));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));  
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    CHECK_CUSPARSE(cusparseXcsr2bsrNnz(chandle, dir, m, n,
                                      descr, d_csrRowPtr, d_csrColInd, blockdim,
                                      descrA, bsrRowPtrA, &nnzb));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrColIndA, sizeof(int)*nnzb));

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&bsrValA, sizeof(float)*(blockdim*blockdim)*nnzb));
    CHECK_CUSPARSE(cusparseScsr2bsr(chandle, dir, m, n,
                                    descr, d_csrVal, d_csrRowPtr, d_csrColInd, blockdim,
                                    descrA, bsrValA, bsrRowPtrA, bsrColIndA));

    int pBufferSize_M;
    void *pBuffer = 0;

    const alphasparseSolvePolicy_t policy_M = ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL;
    
    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    // step 2: create a empty info structure
    // we need one info for bsric02 and two info's for csrsv2
    // step 3: query how much memory used in bsric02 and csrsv2, and allocate the buffer
    alphasparseMatDescr_t descr_M = 0;
    alphasparseCreateMatDescr(&descr_M);
    alphasparseSbsric02_bufferSize(handle, dir_alpha, mb, nnzb,
    descr_M, bsrValA, bsrRowPtrA, bsrColIndA, blockdim, info_M, &pBufferSize_M);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize_M);
    // step 4: perform analysis of incomplete Cholesky on M
    // perform analysis of triangular solve on L
    // perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of bsric02 and csrsv2 simultaneously.
    alphasparseSbsric02_analysis(handle, dir_alpha, mb, nnzb, descr_M,
    bsrValA, bsrRowPtrA, bsrColIndA, blockdim, info_M,
    policy_M, pBuffer);
    // step 5: M = L * L'
    alphasparseSbsric02(handle, dir_alpha, mb, nnzb, descr_M,
    bsrValA, bsrRowPtrA, bsrColIndA, blockdim, info_M, policy_M, pBuffer);
    
    free(ict_val);
    ict_val = (float *)malloc(sizeof(float)*nnzb);
    cudaMemcpy(ict_val, bsrValA, sizeof(float)*nnzb, cudaMemcpyDeviceToHost);
    ict_nnzb = nnzb;
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

  ict_val = (float*)alpha_malloc(nnz * sizeof(float));
  cuda_val = (float*)alpha_malloc(nnz * sizeof(float));

  cuda_ilu02();
  alpha_ilu02();
  check((float*)cuda_val, cuda_nnzb, (float*)ict_val, ict_nnzb);
  for (int i = 0; i < min(20,cuda_nnzb); i++) {
    std::cout << cuda_val[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20,ict_nnzb); i++) {
    std::cout << ict_val[i] << ", ";
  }
  std::cout << std::endl;
  // for (int i = 0; i < ict_nnzb; i++)
  // {
  //   if((cuda_val[i] - ict_val[i]) / cuda_val[i] > 1e-6) 
  //   {
  //     std::cout << "pos " << i << " col indx " << coo_col_index[i] <<" cuda val ("<< std::setprecision(10) << cuda_val[i] << ") ict val (" << std::setprecision(10) << ict_val[i] << ")" << std::endl;
  //   }
  // }
  return 0;
}