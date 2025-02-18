#include "../test_common.h"

/**
 * @brief ict dcu mv csr test
 * @author HPCRC, ICT
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include <alphasparse.h>

const char *file;
bool check_flag;
int iter;

// sparse vector
int nnz = 10000;
int *alpha_x_idx;
int *roc_x_idx;
float *x_val;
float *cuda_y, *alpha_y;
float c = 2., s = 3.;
float roc_res = 0., alpha_res = 0.;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(-1);                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(-1);                                                   \
    }                                                                          \
}

static void roc_spvv() {
  // cusparse handle
  cusparseHandle_t handle;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

  cudaDeviceProp devProp;
  int device_id = 0;

  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&devProp, device_id);
  std::cout << "Device: " << devProp.name << std::endl;

  // Offload data to device
  int *dx_idx = NULL;
  float *dx_val = NULL;
  float *dy = NULL;

  cudaMalloc((void **)&dx_idx, sizeof(int) * nnz);
  cudaMalloc((void **)&dx_val, sizeof(float) * nnz);
  cudaMalloc((void **)&dy, sizeof(float) * nnz * 20);

  cudaMemcpy(dx_idx, roc_x_idx, sizeof(int) * nnz,
            cudaMemcpyHostToDevice);
  cudaMemcpy(dx_val, x_val, sizeof(float) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, cuda_y, sizeof(float) * nnz * 20, cudaMemcpyHostToDevice);

    cusparseSpVecDescr_t x;
    cusparseCreateSpVec(&x, nnz * 20, nnz, (void *)dx_idx,
                                (void *)dx_val, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_R_32F);

    cusparseDnVecDescr_t y;
    cusparseCreateDnVec(&y, nnz * 20, (void *)dy,
                                CUDA_R_32F);
    size_t               bufferSize = 0;
    void*                dBuffer    = NULL;

    CHECK_CUSPARSE( cusparseSpVV_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            x, y, &roc_res, CUDA_R_32F,
                                            &bufferSize) )

    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    // Call cusparse csrmv
    CHECK_CUSPARSE( cusparseSpVV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 x, y, &roc_res, CUDA_R_32F, dBuffer) )

  // Device synchronization
  cudaDeviceSynchronize();

  cudaMemcpy(cuda_y, dy, sizeof(float) * nnz * 20, cudaMemcpyDeviceToHost);

  // Clear up on device
  cudaFree(dx_val);
  cudaFree(dx_idx);
  cudaFree(dy);
  cusparseDestroy(handle);
}

static void alpha_spvv()
{
    // cusparse handle
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    cudaDeviceProp devProp;
    int device_id = 0;

    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    int *dx_idx = NULL;
    float *dx_val     = NULL;
    float *dy         = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * nnz);
    cudaMalloc((void **)&dx_val, sizeof(float) * nnz);
    cudaMalloc((void **)&dy, sizeof(float) * nnz * 20);

    cudaMemcpy(dx_idx, roc_x_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, x_val, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, alpha_y, sizeof(float) * nnz * 20, cudaMemcpyHostToDevice);

    alphasparseSpVecDescr_t x{};
    alphasparseCreateSpVec(&x, nnz * 20,nnz,(void *)dx_idx,(void *)dx_val,ALPHA_SPARSE_INDEXTYPE_I32,ALPHA_SPARSE_INDEX_BASE_ZERO,ALPHA_R_32F);

    alphasparseDnVecDescr_t y{};
    alphasparseCreateDnVec(&y, nnz * 20,(void *)dy,ALPHA_R_32F);

    size_t buffer_size;
    void *temp_buffer;
    alphasparseSpvv_buffersize(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, x, y, (void *)&alpha_res, ALPHA_R_32F, &buffer_size);
    cudaMalloc((void **)&temp_buffer, buffer_size);
    // Call cusparse csrmv
    alphasparseSpvv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, x, y, (void *)&alpha_res, ALPHA_R_32F, temp_buffer);

    
    // Call cusparse csrmv
    alphasparseRot(handle, &c, &s, x, y);

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(alpha_y, dy, sizeof(float) * nnz * 20, cudaMemcpyDeviceToHost);

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    alphasparse_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file  = args_get_data_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    iter  = args_get_iter(argc, argv);
    nnz  = args_get_nnz(argc, argv);

    alpha_x_idx = (int *)alpha_memalign(sizeof(int) * nnz, DEFAULT_ALIGNMENT);
    roc_x_idx   = (int *)alpha_memalign(sizeof(int) * nnz,
                                                DEFAULT_ALIGNMENT);
    x_val       = (float *)alpha_memalign(sizeof(float) * nnz, DEFAULT_ALIGNMENT);
    cuda_y       = (float *)alpha_memalign(sizeof(float) * nnz * 20, DEFAULT_ALIGNMENT);
    alpha_y     = (float *)alpha_memalign(sizeof(float) * nnz * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random(cuda_y, 1, nnz * 20);
    alpha_fill_random(alpha_y, 1, nnz * 20);
    alpha_fill_random(x_val, 0, nnz);

    for (int i = 0; i < nnz; i++) {
        alpha_x_idx[i] = i * 20;
        roc_x_idx[i]   = i * 20;
    }

    alpha_spvv();

    if (check_flag) {
      roc_spvv();
      bool status = fabs(alpha_res - roc_res) > 1e-2;
      if (!status) {
        fprintf(stderr, "doti_s correct, %f\n", fabs(alpha_res - roc_res)/alpha_res);
        fprintf(stderr, "roc : %f, ict : %f\n", roc_res, alpha_res);
      } else {
        fprintf(stderr, "doti_s error\n");
        fprintf(stderr, "roc : %f, ict : %f\n", roc_res, alpha_res);
      }
    }
    printf("\n");

    alpha_free(x_val);
    alpha_free(roc_x_idx);
    alpha_free(alpha_x_idx);
    alpha_free(cuda_y);
    alpha_free(alpha_y);
    return 0;
}
