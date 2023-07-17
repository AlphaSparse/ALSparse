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

int m = 8192, n = 1024, ldb, size;
cuFloatComplex *hdl, *hd, *hdu, *hictB, *hcudaB;
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
  cuFloatComplex* dB = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(cuFloatComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(cuFloatComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(cuFloatComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(cuFloatComplex) * size))

  // Copy data to device
  CHECK_CUDA(
    cudaMemcpy(ddl, hdl, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dd, hd, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(ddu, hdu, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(
    dB, hcudaB, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseCgtsv2_bufferSizeExt(
    handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size))

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));
  CHECK_CUSPARSE(
    cusparseCgtsv2(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer));

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
    hcudaB, dB, sizeof(cuFloatComplex) * size, cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroy(handle))
  CHECK_CUDA(cudaFree(ddl))
  CHECK_CUDA(cudaFree(dd))
  CHECK_CUDA(cudaFree(ddu))
  CHECK_CUDA(cudaFree(dB))
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
  cuFloatComplex* dB = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(cuFloatComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(cuFloatComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(cuFloatComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(cuFloatComplex) * size))

  // Copy data to device
  CHECK_CUDA(
    cudaMemcpy(ddl, hdl, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dd, hd, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(ddu, hdu, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(
    dB, hictB, sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  alphasparseCgtsv2_bufferSizeExt(
    handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size);

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));

  alphasparseCgtsv2(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer);

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
    hictB, dB, sizeof(cuFloatComplex) * size, cudaMemcpyDeviceToHost));

  alphasparse_destory_handle(handle);
  CHECK_CUDA(cudaFree(ddl))
  CHECK_CUDA(cudaFree(dd))
  CHECK_CUDA(cudaFree(ddu))
  CHECK_CUDA(cudaFree(dB))
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
  ldb = m;
  size = ldb * n;
  hdl = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));
  hd = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));
  hdu = (cuFloatComplex*)alpha_malloc(m * sizeof(cuFloatComplex));
  hcudaB = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));
  hictB = (cuFloatComplex*)alpha_malloc(size * sizeof(cuFloatComplex));

  alpha_fill_random(hdl, 899, m);
  hdl[0] = {};
  alpha_fill_random(hd, 101, m);
  alpha_fill_random(hdu, 77, m);
  hdu[m - 1] = {};
  for (size_t i = 0; i < m; ++i)
  {
    if (hd[i].x > 0)
      hd[i].x += 1.0;
    else
      hd[i].x -= 1.0;

    if (hd[i].y > 0)
      hd[i].x += 1.0;
    else
      hd[i].x -= 1.0;
  }
  alpha_fill_random(hcudaB, 1, size);
  alpha_fill_random(hictB, 1, size);
  std::cout << "===========hdl=============" << std::endl;
  for (int i = 0; i < min(20, m); i++) {
    std::cout << hdl[i] << ", ";
  }
  std::cout << std::endl << "===========hdl=============" << std::endl;
  std::cout << std::endl << "===========hd=============" << std::endl;
  for (int i = 0; i < min(20, m); i++) {
    std::cout << hd[i] << ", ";
  }
  std::cout << std::endl << "===========hd=============" << std::endl;
  std::cout << std::endl << "===========hdu=============" << std::endl;
  for (int i = 0; i < min(20, m); i++) {
    std::cout << hdu[i] << ", ";
  }
  std::cout << std::endl << "===========hdu=============" << std::endl;
  std::cout << std::endl << "===========hcudaB=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcudaB[i] << ", ";
  }
  std::cout << std::endl << "===========hcudaB=============" << std::endl;
  std::cout << std::endl << "===========hictB=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hictB[i] << ", ";
  }
  std::cout << std::endl << "===========hictB=============" << std::endl;

  cuda_gtsv2();
  alpha_gtsv2();

  std::cout << std::endl << "===========result=============" << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcudaB[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hictB[i] << ", ";
  }
  std::cout << std::endl << "===========result=============" << std::endl;
  check(hictB, size, hcudaB, size, &error);

  return 0;
}
