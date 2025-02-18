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

int m = 1024, n = 1024, ldb, size;
cuDoubleComplex *hdl, *hd, *hdu, *hictB, *hcudaB;

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

  cuDoubleComplex* ddl = NULL;
  cuDoubleComplex* dd = NULL;
  cuDoubleComplex* ddu = NULL;
  cuDoubleComplex* dB = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(cuDoubleComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(cuDoubleComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(cuDoubleComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(cuDoubleComplex) * size))

  // Copy data to device
  CHECK_CUDA(
    cudaMemcpy(ddl, hdl, sizeof(cuDoubleComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dd, hd, sizeof(cuDoubleComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(ddu, hdu, sizeof(cuDoubleComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(
    dB, hcudaB, sizeof(cuDoubleComplex) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseZgtsv2_nopivot_bufferSizeExt(
    handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size))

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));
  CHECK_CUSPARSE(
    cusparseZgtsv2_nopivot(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer));

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
    hcudaB, dB, sizeof(cuDoubleComplex) * size, cudaMemcpyDeviceToHost));

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

  cuDoubleComplex* ddl = NULL;
  cuDoubleComplex* dd = NULL;
  cuDoubleComplex* ddu = NULL;
  cuDoubleComplex* dB = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(cuDoubleComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(cuDoubleComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(cuDoubleComplex) * m))
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(cuDoubleComplex) * size))

  // Copy data to device
  CHECK_CUDA(
    cudaMemcpy(ddl, hdl, sizeof(cuDoubleComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dd, hd, sizeof(cuDoubleComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(ddu, hdu, sizeof(cuDoubleComplex) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(
    dB, hictB, sizeof(cuDoubleComplex) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  alphasparseZgtsv2_nopivot_bufferSizeExt(
    handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size);

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));

  alphasparseZgtsv2_nopivot(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer);

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
    hictB, dB, sizeof(cuDoubleComplex) * size, cudaMemcpyDeviceToHost));

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
  hdl = (cuDoubleComplex*)alpha_malloc(m * sizeof(cuDoubleComplex));
  hd = (cuDoubleComplex*)alpha_malloc(m * sizeof(cuDoubleComplex));
  hdu = (cuDoubleComplex*)alpha_malloc(m * sizeof(cuDoubleComplex));
  hcudaB = (cuDoubleComplex*)alpha_malloc(size * sizeof(cuDoubleComplex));
  hictB = (cuDoubleComplex*)alpha_malloc(size * sizeof(cuDoubleComplex));

  alpha_fill_random(hdl, 55, m);
  hdl[0] = cuDoubleComplex{};
  alpha_fill_random(hd, 99, m);
  alpha_fill_random(hdu, 1, m);
  hdu[m - 1] = cuDoubleComplex{};
  alpha_fill_random(hcudaB, 1, size);
  alpha_fill_random(hictB, 1, size);
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcudaB[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hictB[i] << ", ";
  }
  std::cout << std::endl;
  cuda_gtsv2();
  alpha_gtsv2();

  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcudaB[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hictB[i] << ", ";
  }
  std::cout << std::endl;

  check(hictB, size, hcudaB, size);

  return 0;
}
