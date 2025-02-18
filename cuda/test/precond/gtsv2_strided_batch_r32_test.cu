#include "../test_common.h"

/**
 * @brief ict dcu mv hyb test
 * @author HPCRC, ICT
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "alphasparse.h"

bool check_flag;

alphasparseOperation_t transA;
alphasparseDirection_t dir_alpha;

int m = 32, n = 16, batchCount, batchStride, size;
float *hdl, *hd, *hdu, *hict_x, *hcuda_x;

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

void
cuda_gtsv2()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  float* ddl = NULL;
  float* dd = NULL;
  float* ddu = NULL;
  float* dx = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(float) * size))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(float) * size))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(float) * size))
  CHECK_CUDA(cudaMalloc((void**)&dx, sizeof(float) * size))

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(ddl, hdl, sizeof(float) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dd, hd, sizeof(float) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(ddu, hdu, sizeof(float) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dx, hcuda_x, sizeof(float) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseSgtsv2StridedBatch_bufferSizeExt(
    handle, m, ddl, dd, ddu, dx, batchCount, batchStride, &buffer_size))

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));
  CHECK_CUSPARSE(cusparseSgtsv2StridedBatch(
    handle, m, ddl, dd, ddu, dx, batchCount, batchStride, temp_buffer));

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
    cudaMemcpy(hcuda_x, dx, sizeof(float) * size, cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroy(handle))
  CHECK_CUDA(cudaFree(ddl))
  CHECK_CUDA(cudaFree(dd))
  CHECK_CUDA(cudaFree(ddu))
  CHECK_CUDA(cudaFree(dx))
  CHECK_CUDA(cudaFree(temp_buffer))
}

void
alpha_gtsv2()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  float* ddl = NULL;
  float* dd = NULL;
  float* ddu = NULL;
  float* dx = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(float) * size))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(float) * size))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(float) * size))
  CHECK_CUDA(cudaMalloc((void**)&dx, sizeof(float) * size))

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(ddl, hdl, sizeof(float) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dd, hd, sizeof(float) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(ddu, hdu, sizeof(float) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dx, hict_x, sizeof(float) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  alphasparseSgtsv2StridedBatch_bufferSizeExt(
    handle, m, ddl, dd, ddu, dx, batchCount, batchStride, &buffer_size);

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));

  alphasparseSgtsv2StridedBatch(
    handle, m, ddl, dd, ddu, dx, batchCount, batchStride, temp_buffer);

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
    cudaMemcpy(hict_x, dx, sizeof(float) * size, cudaMemcpyDeviceToHost));

  alphasparse_destory_handle(handle);
  CHECK_CUDA(cudaFree(ddl))
  CHECK_CUDA(cudaFree(dd))
  CHECK_CUDA(cudaFree(ddu))
  CHECK_CUDA(cudaFree(dx))
  CHECK_CUDA(cudaFree(temp_buffer))
}

int
main(int argc, const char* argv[])
{
  // args
  args_help(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  m = args_get_rows(argc, argv, m);
  n = args_get_cols(argc, argv, n);
  batchStride = m;
  batchCount = n;
  size = batchCount * batchStride;
  hdl = (float*)alpha_malloc(size * sizeof(float));
  hd = (float*)alpha_malloc(size * sizeof(float));
  hdu = (float*)alpha_malloc(size * sizeof(float));
  hcuda_x = (float*)alpha_malloc(size * sizeof(float));
  hict_x = (float*)alpha_malloc(size * sizeof(float));

  alpha_fill_random(hdl, 899, size);
  alpha_fill_random(hd, 101, size);
  alpha_fill_random(hdu, 77, size);
  alpha_fill_random(hcuda_x, 1, size);
  alpha_fill_random(hict_x, 1, size);
  for(int j = 0; j < batchCount; ++j)
  {
      hdl[j * batchStride + 0]     = {};
      hdu[j * batchStride + m - 1] = {};
  }
  std::cout << "===========hdl=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hdl[i] << ", ";
  }
  std::cout << std::endl << "===========hdl=============" << std::endl;
  std::cout << std::endl << "===========hd=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hd[i] << ", ";
  }
  std::cout << std::endl << "===========hd=============" << std::endl;
  std::cout << std::endl << "===========hdu=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hdu[i] << ", ";
  }
  std::cout << std::endl << "===========hdu=============" << std::endl;
  std::cout << std::endl << "===========hcuda_x=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcuda_x[i] << ", ";
  }
  std::cout << std::endl << "===========hcuda_x=============" << std::endl;
  std::cout << std::endl << "===========hict_x=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hict_x[i] << ", ";
  }
  std::cout << std::endl << "===========hict_x=============" << std::endl;

  cuda_gtsv2();
  alpha_gtsv2();

  std::cout << std::endl << "===========result=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcuda_x[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hict_x[i] << ", ";
  }
  std::cout << std::endl << "===========result=============" << std::endl;
  check(hict_x, size, hcuda_x, size);

  return 0;
}
