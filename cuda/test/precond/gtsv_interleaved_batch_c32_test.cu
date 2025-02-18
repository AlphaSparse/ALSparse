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

int algo = 0;
int m = 128, batchCount = 8, batchStride = 32, size;
cuFloatComplex *hdl, *hd, *hdu, *hict_x, *hcuda_x;
float error;

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

  cuFloatComplex* ddl = NULL;
  cuFloatComplex* dd = NULL;
  cuFloatComplex* ddu = NULL;
  cuFloatComplex* dx = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(cuFloatComplex) * size))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(cuFloatComplex) * size))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(cuFloatComplex) * size))
  CHECK_CUDA(cudaMalloc((void**)&dx, sizeof(cuFloatComplex) * size))

  // Copy data to device
  CHECK_CUDA(
    cudaMemcpy(ddl, hdl, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dd, hd, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(ddu, hdu, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(
    dx, hcuda_x, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseCgtsvInterleavedBatch_bufferSizeExt(
    handle, algo, m, ddl, dd, ddu, dx, batchCount, &buffer_size))

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));
  CHECK_CUSPARSE(cusparseCgtsvInterleavedBatch(
    handle, algo, m, ddl, dd, ddu, dx, batchCount, temp_buffer));

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
    hcuda_x, dx, sizeof(cuFloatComplex) * size, cudaMemcpyDeviceToHost));

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

  cuFloatComplex* ddl = NULL;
  cuFloatComplex* dd = NULL;
  cuFloatComplex* ddu = NULL;
  cuFloatComplex* dx = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(cuFloatComplex) * size))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(cuFloatComplex) * size))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(cuFloatComplex) * size))
  CHECK_CUDA(cudaMalloc((void**)&dx, sizeof(cuFloatComplex) * size))

  // Copy data to device
  CHECK_CUDA(
    cudaMemcpy(ddl, hdl, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dd, hd, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(ddu, hdu, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(
    dx, hict_x, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  alphasparseCgtsvInterleavedBatch_bufferSizeExt(
    handle, algo, m, ddl, dd, ddu, dx, batchCount, &buffer_size);

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));

  alphasparseCgtsvInterleavedBatch(
    handle, algo, m, ddl, dd, ddu, dx, batchCount, temp_buffer);

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
    hict_x, dx, sizeof(cuFloatComplex) * size, cudaMemcpyDeviceToHost));

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
  batchCount = args_get_batch_count(argc, argv, batchCount);
  batchStride = args_get_batch_stride(argc, argv, batchStride);
  size = m * batchCount;

  hdl = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));
  hd = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));
  hdu = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));
  hcuda_x = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));
  hict_x = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));

  alpha_fill_random(hdl, 899, size);
  alpha_fill_random(hd, 101, size);
  alpha_fill_random(hdu, 77, size);
  for (size_t i = 0; i < m; ++i)
  {
    if (hd[i].x > 0)
      hd[i].x += m/2;
    else
      hd[i].x -= m/2;

    if (hd[i].y > 0)
      hd[i].y += m/2;
    else
      hd[i].y -= m/2;
  }
  for(int i = 0; i < batchCount; i++)
    hdl[i * m] = {};
  for(int i = 0; i < batchCount; i++)
    hdu[i* m - 1] = {};
  alpha_fill_random(hcuda_x, 1, size);
  alpha_fill_random(hict_x, 1, size);
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
  check(hict_x, size, hcuda_x, size, &error);

  return 0;
}
