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

int m, n, nnz;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
cuFloatComplex* coo_values;

// coo format
cuFloatComplex* x_val;
cuFloatComplex* ict_y;
cuFloatComplex* cuda_y;

cuFloatComplex* ict_val;
cuFloatComplex* cuda_val;

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

void cuda_ilu02()
{
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Offload data to device
    int* d_csrRowPtr = NULL;
    int* dArow = NULL;
    int* d_csrColInd = NULL;
    cuFloatComplex* d_csrVal = NULL;
    cuFloatComplex* d_x = NULL;
    cuFloatComplex* d_y = NULL;
    cuFloatComplex* d_z = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, m * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_z, m * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMemcpy(d_x, x_val, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, cuda_y, m * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrVal, sizeof(cuFloatComplex) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (m + 1)));
    
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_csrColInd, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_csrVal, coo_values, nnz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, d_csrRowPtr);
    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    csrilu02Info_t info_M = 0;

    int pBufferSize_M;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;
    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));
    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&info_M));
    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    CHECK_CUSPARSE(cusparseCcsrilu02_bufferSize(handle, m, nnz,
    descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M));
    pBufferSize = pBufferSize_M;
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);
    // step 4: perform analysis of incomplete Cholesky on M
    // perform analysis of triangular solve on L
    // perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csrilu02 and csrsv2 simultaneously.
    CHECK_CUSPARSE(cusparseCcsrilu02_analysis(handle, m, nnz, descr_M,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
    policy_M, pBuffer));
    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    // step 5: M = L * L'
    CHECK_CUSPARSE(cusparseCcsrilu02(handle, m, nnz, descr_M,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer));
    status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
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
    cudaMemcpy(cuda_val, d_csrVal, sizeof(cuFloatComplex)*nnz, cudaMemcpyDeviceToHost);
    // step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyCsrilu02Info(info_M);
    cusparseDestroy(handle);
}

void alpha_ilu02()
{
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    // Offload data to device
    int* d_csrRowPtr = NULL;
    int* dArow = NULL;
    int* d_csrColInd = NULL;
    cuFloatComplex* d_csrVal = NULL;
    cuFloatComplex* d_x = NULL;
    cuFloatComplex* d_y = NULL;
    cuFloatComplex* d_z = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, m * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_z, m * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMemcpy(d_x, x_val, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, cuda_y, m * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrVal, sizeof(cuFloatComplex) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (m + 1)));
    
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_csrColInd, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_csrVal, coo_values, nnz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, d_csrRowPtr);
    alphasparseMatDescr_t descr_M = 0;
    alphasparseMatDescr_t descr_L = 0;
    alpha_csrilu02Info_t info_M = ALPHA_SPARSE_OPAQUE;

    int pBufferSize_M;
    size_t pBufferSize_L;
    size_t pBufferSize_Lt;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const alphasparseSolvePolicy_t policy_M = ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL;
    const alphasparseSolvePolicy_t policy_L = ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL;
    const alphasparseSolvePolicy_t policy_Lt = ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL;
    const alphasparseOperation_t trans_L = ALPHA_SPARSE_OPERATION_NON_TRANSPOSE;
    const alphasparseOperation_t trans_Lt = ALPHA_SPARSE_OPERATION_TRANSPOSE;
   
    // alphasparseSpMatDescr_t csr;
    // alphasparseCreateCsr(&csr,
    //                     m,
    //                     n,
    //                     nnz,
    //                     d_csrRowPtr,
    //                     d_csrColInd,
    //                     d_csrVal,
    //                     ALPHA_SPARSE_INDEXTYPE_I32,
    //                     ALPHA_SPARSE_INDEXTYPE_I32,
    //                     ALPHA_SPARSE_INDEX_BASE_ZERO,
    //                     ALPHA_R_32F);
      
    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    alphasparseCreateMatDescr(&descr_M);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    alphasparseCcsrilu02_bufferSize(handle, m, nnz,
    descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
    alphasparseDnVecDescr_t x{};
    alphasparseCreateDnVec(&x, n, (void*)d_x, ALPHA_R_32F);

    alphasparseDnVecDescr_t y_ict{};
    alphasparseCreateDnVec(&y_ict, m, (void*)d_y, ALPHA_R_32F);
    alphasparseDnVecDescr_t d_zz{};
    alphasparseCreateDnVec(&d_zz, m, (void*)d_z, ALPHA_R_32F);

    alphasparseSpSVDescr_t spsvDescr;
    alphasparseSpSV_createDescr(&spsvDescr);
    // alphasparseSpSV_bufferSize(handle,
    //                           ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
    //                           &alpha,
    //                           csr,
    //                           x,
    //                           y_ict,
    //                           ALPHA_R_32F,
    //                           ALPHA_SPARSE_SPSV_ALG_DEFAULT,
    //                           spsvDescr,
    //                           &pBufferSize_L);

    // alphasparseSpSV_bufferSize(handle,
    //                           ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
    //                           &alpha,
    //                           csr,
    //                           x,
    //                           y_ict,
    //                           ALPHA_R_32F,
    //                           ALPHA_SPARSE_SPSV_ALG_DEFAULT,
    //                           spsvDescr,
    //                           &pBufferSize_Lt);

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize_M);
    // step 4: perform analysis of incomplete Cholesky on M
    // perform analysis of triangular solve on L
    // perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csrilu02 and csrsv2 simultaneously.
    alphasparseCcsrilu02_analysis(handle, m, nnz, descr_M,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
    policy_M, pBuffer);
    // step 5: M = L * L'
    alphasparseCcsrilu02(handle, m, nnz, descr_M,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);

    cudaMemcpy(ict_val, d_csrVal, sizeof(cuFloatComplex)*nnz, cudaMemcpyDeviceToHost);
//     alphasparseSpMatDescr_t csr_L;
//     alphasparseCreateCsr(&csr_L,
//                         m,
//                         n,
//                         nnz,
//                         d_csrRowPtr,
//                         d_csrColInd,
//                         d_csrVal,
//                         ALPHA_SPARSE_INDEXTYPE_I32,
//                         ALPHA_SPARSE_INDEXTYPE_I32,
//                         ALPHA_SPARSE_INDEX_BASE_ZERO,
//                         ALPHA_R_32F);

//     void* dBuffer = NULL;
//     size_t bufferSize = 0;
//     alphasparse_fill_mode_t fillmode = ALPHA_SPARSE_FILL_MODE_LOWER;
//     alphasparseSpMatSetAttribute(
//       csr_L, ALPHASPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
//     // Specify Unit|Non-Unit diagonal type.
//     alphasparse_diag_type_t diagtype = ALPHA_SPARSE_DIAG_NON_UNIT;
//     alphasparseSpMatSetAttribute(
//       csr_L, ALPHASPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
//     alphasparseSpSV_bufferSize(handle,
//                               ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
//                               &alpha,
//                               csr_L,
//                               x,
//                               y_ict,
//                               ALPHA_R_32F,
//                               ALPHA_SPARSE_SPSV_ALG_DEFAULT,
//                               spsvDescr,
//                               &bufferSize);

//     CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

//     alphasparseSpSV_analysis(handle,
//                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
//                             &alpha,
//                             csr_L,
//                             x,
//                             y_ict,
//                             ALPHA_R_32F,
//                             ALPHA_SPARSE_SPSV_ALG_DEFAULT,
//                             spsvDescr,
//                             dBuffer);
                            
//     // step 6: solve L*z = x
//     alphasparseSpSV_solve(handle,
//                           ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
//                           &alpha,
//                           csr_L,
//                           x,
//                           d_zz,
//                           ALPHA_R_32F,
//                           ALPHA_SPARSE_SPSV_ALG_DEFAULT,
//                           spsvDescr);
//     // step 7: solve L'*y = z             
//     alphasparseSpSV_solve(handle,
//                           ALPHA_SPARSE_OPERATION_TRANSPOSE,
//                           &alpha,
//                           csr_L,
//                           d_zz,
//                           y_ict,
//                           ALPHA_R_32F,
//                           ALPHA_SPARSE_SPSV_ALG_DEFAULT,
//                           spsvDescr);   
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
  alpha_read_coo<cuFloatComplex>(
    file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, cuFloatComplex>(nnz, coo_row_index, coo_col_index, coo_values);
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
  x_val = (cuFloatComplex*)alpha_malloc(n * sizeof(cuFloatComplex));
  ict_y = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));
  cuda_y = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));

  ict_val = (cuFloatComplex*)alpha_malloc(nnz * sizeof(cuFloatComplex));
  cuda_val = (cuFloatComplex*)alpha_malloc(nnz * sizeof(cuFloatComplex));

  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  cuda_ilu02();
  alpha_ilu02();
  check((cuFloatComplex*)cuda_val, nnz, (cuFloatComplex*)ict_val, nnz);
  for (int i = 0; i < min(20,nnz); i++) {
    std::cout << cuda_val[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20,nnz); i++) {
    std::cout << ict_val[i] << ", ";
  }
  std::cout << std::endl;
  return 0;
}