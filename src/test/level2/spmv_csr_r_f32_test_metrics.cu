
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
#include <chrono>

const char *file, *metrics_file;
int thread_num;
bool check_flag;
bool metrics_flag;
int iter, warm_up = 0, trials = 1;
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
const float alpha = 2.1f;
const float beta = 3.2f;

std::vector<double> cuda_time_list, alpha_time_list, cuda_bandwidth_list, alpha_bandwidth_list, cuda_gflops_list, alpha_gflops_list;
std::vector<cusparseSpMVAlg_t> cu_alg_list = {CUSPARSE_SPMV_ALG_DEFAULT, CUSPARSE_SPMV_CSR_ALG1, CUSPARSE_SPMV_CSR_ALG2};
std::vector<alphasparseSpMVAlg_t> alpha_alg_list = {ALPHA_SPARSE_SPMV_ROW_PARTITION};
// std::vector<cusparseSpMVAlg_t> cu_alg_list = {CUSPARSE_SPMV_ALG_DEFAULT};
// std::vector<alphasparseSpMVAlg_t> alpha_alg_list = {ALPHA_SPARSE_SPMV_ADAPTIVE};

cudaEvent_t event_start, event_stop;
float elapsed_time = 0.0;

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
  auto start_time = std::chrono::high_resolution_clock::now();
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
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
  std::cout << "cusparseCreateCsr: " << duration.count() << " ns" << std::endl;

  size_t bufferSize = 0;
  void *dBuffer = NULL;
  for (auto alg : cu_alg_list)
  {
    // double time2 = get_time_us();
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA,
                                           vecX,
                                           &beta,
                                           vecY,
                                           CUDA_R_32F,
                                           alg,
                                           &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    // time2 = (get_time_us() - time2) / (1e3);
    // printf("\n==========================\n");
    // printf("cu pre_time:%lf", time2);
    // printf("\n==========================\n");
    for (int i = 0; i < warm_up; ++i)
    {
      // double time2 = get_time_us();
      CHECK_CUSPARSE(cusparseSpMV(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha,
                                  matA,
                                  vecX,
                                  &beta,
                                  vecY,
                                  CUDA_R_32F,
                                  alg,
                                  dBuffer));
      cudaDeviceSynchronize();
      // time2 = (get_time_us() - time2) / (1e3);
      // printf("\n==========================\n");
      // printf("warmup_time:%lf",time2);
      // printf("\n==========================\n");
    }
    std::vector<double> times;
    for (int i = 0; i < trials; ++i)
    {
      CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(float), cudaMemcpyHostToDevice));
      GPU_TIMER_START(elapsed_time, event_start, event_stop);
      CHECK_CUSPARSE(cusparseSpMV(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha,
                                  matA,
                                  vecX,
                                  &beta,
                                  vecY,
                                  CUDA_R_32F,
                                  alg,
                                  dBuffer));
      cudaDeviceSynchronize();
      GPU_TIMER_END(elapsed_time, event_start, event_stop);
      // printf("\ncompute_time:%lf", time);
      times.push_back(elapsed_time);
    }
    double time = get_avg_time(times);
    printf("cusparse %d: %lf ms\n", alg, time);
    double bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + sizeof(int) * (m + 1 + nnz) / time / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;
    cuda_time_list.push_back(time);
    cuda_bandwidth_list.push_back(bandwidth);
    cuda_gflops_list.push_back(gflops);
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
  auto start_time = std::chrono::high_resolution_clock::now();
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
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
  std::cout << "alphasparseCreateCsr: " << duration.count() << " ns" << std::endl;

  void *dBuffer = NULL;
  size_t bufferSize = 0;
  for (auto alg : alpha_alg_list)
  {
    alphasparseSpMV_bufferSize(handle,
                               ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha,
                               csr,
                               x,
                               &beta,
                               y_ict,
                               ALPHA_R_32F,
                               alg,
                               &bufferSize);
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
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
                      alg,
                      dBuffer);
      cudaDeviceSynchronize();
    }
    std::vector<double> times;
    for (int i = 0; i < trials; ++i)
    {
      CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(float), cudaMemcpyHostToDevice));
      GPU_TIMER_START(elapsed_time, event_start, event_stop);
      alphasparseSpMV(handle,
                      ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha,
                      csr,
                      x,
                      &beta,
                      y_ict,
                      ALPHA_R_32F,
                      alg,
                      dBuffer);
      cudaDeviceSynchronize();
      GPU_TIMER_END(elapsed_time, event_start, event_stop);
      times.push_back(elapsed_time);
    }
    double time = get_avg_time(times);
    printf("alphasparse %d: %lf ms\n", alg, time);
    // double bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + sizeof(int) * (m + 1 + nnz) / time / 1e6;
    // double gflops = static_cast<double>(2 * nnz) / time / 1e6;
    // alpha_time_list.push_back(time);
    // alpha_bandwidth_list.push_back(bandwidth);
    // alpha_gflops_list.push_back(gflops);
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
  const char *libname = args_get_libname(argc, argv);
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

  // init x y
  x_val = (float *)alpha_malloc(n * sizeof(float));
  ict_y = (float *)alpha_malloc(m * sizeof(float));
  cuda_y = (float *)alpha_malloc(m * sizeof(float));

  alpha_fill_random(x_val, 2, n);
  alpha_fill_random(ict_y, 1, m);
  alpha_fill_random(cuda_y, 1, m);
  // for (int i = 0; i < 20; i++)
  // {
  //   std::cout << ict_y[i] << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 20; i++)
  // {
  //   std::cout << cuda_y[i] << ", ";
  // }
  printf("\n");
  warm_up = 100;
  trials = 100;
  cuda_mv();
  alpha_mv();
  // std::ofstream filename(metrics_file, std::ios::app);
  // for (size_t i = 0; i < cu_alg_list.size(); i++)
  // {
  //   filename << "Parameters:LIB=\"cuSPARSE\",LEVEL=2,FUNCTIONS=\"SpMV\",FORMAT=\"CSR\",OPERATION=\"N_TRANS\",ALGO=" << cu_alg_list[i] << ",A DATATYPE=\"R_32F\",X DATATYPE=\"R_32F\",Y DATATYPE=\"R_32F\",COMPUTE=\"R_32F\"\n";
  //   filename << "Results:TEST Mat=" << file << ",time=" << cuda_time_list[i] << ",Perf=" << cuda_gflops_list[i] << "\n";
  // }
  // for (size_t i = 0; i < alpha_alg_list.size(); i++)
  // {
  //   filename << "Parameters:LIB=\"AlphaSparse\",LEVEL=2,FUNCTIONS=\"SpMV\",FORMAT=\"CSR\",OPERATION=\"N_TRANS\",ALGO=" << alpha_alg_list[i] << ",A DATATYPE=\"R_32F\",X DATATYPE=\"R_32F\",Y DATATYPE=\"R_32F\",COMPUTE=\"R_32F\"\n";
  //   filename << "Results:TEST Mat=" << file << ",time=" << alpha_time_list[i] << ",Perf=" << alpha_gflops_list[i] << "\n";
  // }
  // filename.close();
  // for (int i = 0; i < 20; i++)
  // {
  //   std::cout << ict_y[i] - 1 << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < 20; i++)
  // {
  //   std::cout << cuda_y[i] - 1 << ", ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < m; i++)
  // {
  //   if (fabs(cuda_y[i] - ict_y[i]) / ict_y[i] > 1e-4)
  //     std::cout << std::fixed << std::setprecision(6) << cuda_y[i] << "," << ict_y[i] << "\n";
  // }
  // std::cout << std::endl;

  check((float *)cuda_y, m, (float *)ict_y, m);
  return 0;
}
