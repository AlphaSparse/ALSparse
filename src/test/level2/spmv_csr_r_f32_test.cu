
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

const char *file, *metrics_file;
int thread_num;
bool check_flag;
bool metrics_flag;
int iter, warm_up = 0, trials = 1;
float error;
alphasparseOperation_t transA;

int m, n, nnz;
int *csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
float *coo_values;

// coo format
float *x_val;
float *ict_y;
float *cuda_y;

// parms for kernel
const float alpha = 2.f;
const float beta = 3.f;

std::vector<double> cuda_times, alpha_times;

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
cuda_mv()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  float *dX = NULL;
  float *dY = NULL;
  int *dCsrRowPtr = NULL;
  int *dArow = NULL;
  int *dAcol = NULL;
  float *dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dCsrRowPtr, sizeof(int) * (m + 1)));
  CHECK_CUDA(cudaMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);
  cusparseDnVecDescr_t vecX, vecY;
  cusparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void **)&dX, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(float), cudaMemcpyHostToDevice));
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, dX, CUDA_R_32F));
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, dY, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                   m,
                                   n,
                                   nnz,
                                   dCsrRowPtr,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         vecX,
                                         &beta,
                                         vecY,
                                         CUDA_R_32F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         &bufferSize));
  void *dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  for (int i = 0; i < warm_up; ++i)
  {
    CHECK_CUSPARSE(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                matA,
                                vecX,
                                &beta,
                                vecY,
                                CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));
    cudaDeviceSynchronize();
  }

  for (int i = 0; i < trials; ++i)
  {
    double time = get_time_us();
    CHECK_CUSPARSE(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                matA,
                                vecX,
                                &beta,
                                vecY,
                                CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));
    cudaDeviceSynchronize();
    time = (get_time_us() - time) / (1e3);
    cuda_times.push_back(time);
  }
  CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dX);
  cudaFree(dY);
  cusparseDestroy(handle);
}

static void
alpha_mv()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  float *dX = NULL;
  float *dY = NULL;
  int *dCsrRowPtr = NULL;
  int *dArow = NULL;
  int *dAcol = NULL;
  float *dAval = NULL;

  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void **)&dCsrRowPtr, sizeof(int) * (m + 1)));

  CHECK_CUDA(cudaMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dAval, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);

  alphasparseDnVecDescr_t vecX, vecY;
  alphasparseSpMatDescr_t matA;
  CHECK_CUDA(cudaMalloc((void **)&dX, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dY, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(float), cudaMemcpyHostToDevice));

  alphasparseDnVecDescr_t x{};
  alphasparseCreateDnVec(&x, n, (void *)dX, ALPHA_R_32F);

  alphasparseDnVecDescr_t y_ict{};
  alphasparseCreateDnVec(&y_ict, m, (void *)dY, ALPHA_R_32F);

  alphasparseSpMatDescr_t csr;
  alphasparseCreateCsr(&csr,
                       m,
                       n,
                       nnz,
                       dCsrRowPtr,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_32F);
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  alphasparseSpMV_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             csr,
                             x,
                             &beta,
                             y_ict,
                             ALPHA_R_32F,
                             ALPHA_SPARSE_SPMV_ALG_COO,
                             &bufferSize);
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  for (int i = 0; i < warm_up; ++i)
  {
    alphasparseSpMV(handle,
                    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    csr,
                    x,
                    &beta,
                    y_ict,
                    ALPHA_R_32F,
                    ALPHA_SPARSE_SPMV_ALG_COO,
                    dBuffer);
    cudaDeviceSynchronize();
  }

  for (int i = 0; i < trials; ++i)
  {
    double time = get_time_us();
    alphasparseSpMV(handle,
                    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    csr,
                    x,
                    &beta,
                    y_ict,
                    ALPHA_R_32F,
                    ALPHA_SPARSE_SPMV_ALG_COO,
                    dBuffer);
    cudaDeviceSynchronize();
    time = (get_time_us() - time) / (1e3);
    alpha_times.push_back(time);
  }
  CHECK_CUDA(cudaMemcpy(ict_y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
}

int main(int argc, const char *argv[])
{
  // args
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  metrics_file = args_save_metrics_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  metrics_flag = args_get_if_calculate_metrics(argc, argv);
  transA = alpha_args_get_transA(argc, argv);

  // read coo
  alpha_read_coo<float>(
      file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, float>(nnz, coo_row_index, coo_col_index, coo_values);
  csrRowPtr = (int *)alpha_malloc(sizeof(int) * (m + 1));
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
  {
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
  x_val = (float *)alpha_malloc(n * sizeof(float));
  ict_y = (float *)alpha_malloc(m * sizeof(float));
  cuda_y = (float *)alpha_malloc(m * sizeof(float));

  alpha_fill_random(x_val, 0, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  if (metrics_flag)
  {
    warm_up = 1;
    trials = 3;
  }
  else if (!metrics_flag && check_flag)
  {
    warm_up = 0;
    trials = 1;
  }
  alpha_mv();
  if (check_flag || metrics_flag)
  {
    cuda_mv();
    if (metrics_flag)
    {

      // 打开文件，如果文件不存在则创建一个新文件
      std::ofstream filename(metrics_file, std::ios::app);
      int check_pass = check((float *)cuda_y, m, (float *)ict_y, m);
      char *if_pass = "";
      if (check_pass == 0)
      {
        if_pass = "PASS";
      }
      else
      {
        if_pass = "FAIL";
      }
      double time = get_avg_time(alpha_times);
      double bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + sizeof(int) * (m + 1 + nnz) / time / 1e6;
      double gflops = static_cast<double>(2 * nnz) / time / 1e6;
      filename << "Parameters:LIB=\"AlphaSparse\",FUNCTIONS=\"SpMV\",FORMAT=\"CSR\",OPERATION=\"N_TRANS\",ALGO=\"ADAPTIVE\",A DATATYPE=\"R_32F\",X DATATYPE=\"R_32F\",Y DATATYPE=\"R_32F\",COMPUTE=\"R_32F\"\n";
      filename << "Results:TEST Mat=" << file << ",Residual=" << error << ",CHECK=" << if_pass << ",time=" << time << ",Perf=" << gflops << "\n";
      time = get_avg_time(cuda_times);
      bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + sizeof(int) * (m + 1 + nnz) / time / 1e6;
      gflops = static_cast<double>(2 * nnz) / time / 1e6;
      filename << "Parameters:LIB=\"cuSPARSE\",FUNCTIONS=\"SpMV\",FORMAT=\"CSR\",OPERATION=\"N_TRANS\",ALGO=\"DEFAULT\",A DATATYPE=\"R_32F\",X DATATYPE=\"R_32F\",Y DATATYPE=\"R_32F\",COMPUTE=\"R_32F\"\n";
      filename << "Results:TEST Mat=" << file << ",Residual=0,CHECK=PASS,time=" << time << ",Perf=" << gflops << "\n";
      filename.close();
    }
    else
    {
      check((float *)cuda_y, m, (float *)ict_y, m);
    }
  }

  // for (int i = 0; i < 20; i++) {
  //   std::cout << cuda_y[i] << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << ict_y[i] << ", ";
  // }
  return 0;
}
