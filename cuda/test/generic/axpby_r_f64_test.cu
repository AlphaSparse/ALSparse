#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "alphasparse.h"

#include "../test_common.h"

const char *file;
bool check_flag;
int iter;

// sparse vector
int nnz;
int *alpha_x_idx;
int *cuda_x_idx;
double *x_val, *cuda_y, *alpha_y, *cpu_y;
double alpha = 2.;
double beta = 3.;

int idx_n = 1000;

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

static void cuda_axpby() {
    cudaDeviceProp devProp;
    int device_id = 0;

    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    int *dx_idx = NULL;
    double *dx_val = NULL;
    double *dy = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * idx_n);
    cudaMalloc((void **)&dx_val, sizeof(double) * idx_n);
    cudaMalloc((void **)&dy, sizeof(double) * idx_n * 20);

    cudaMemcpy(dx_idx, cuda_x_idx, sizeof(int) * idx_n,
            cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, x_val, sizeof(double) * idx_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, cuda_y, sizeof(double) * idx_n * 20, cudaMemcpyHostToDevice);

    // cudaSPARSE handle
    cusparseHandle_t     handle = NULL;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    cusparseSpVecDescr_t x;
    cusparseCreateSpVec(&x, idx_n * 20, idx_n, (void *)dx_idx,
                                (void *)dx_val, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_R_64F);

    cusparseDnVecDescr_t y;
    cusparseCreateDnVec(&y, idx_n * 20, (void *)dy,
                                CUDA_R_64F);

    // Call cudasparse csrmv
    roc_call_exit(cusparseAxpby(handle, (void *)&alpha, x, (void *)&beta, y),
                "cudasparse_axpby");

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_y, dy, sizeof(double) * idx_n * 20, cudaMemcpyDeviceToHost);

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    cusparseDestroy(handle);
}

static void alpha_axpby()
{
    // cudaSPARSE handle
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
    double *dx_val     = NULL;
    double *dy         = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * idx_n);
    cudaMalloc((void **)&dx_val, sizeof(double) * idx_n);
    cudaMalloc((void **)&dy, sizeof(double) * idx_n * 20);

    cudaMemcpy(dx_idx, alpha_x_idx, sizeof(int) * idx_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, x_val, sizeof(double) * idx_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, alpha_y, sizeof(double) * idx_n * 20, cudaMemcpyHostToDevice);

    alphasparseSpVecDescr_t x{};
    alphasparseCreateSpVec(&x,idx_n * 20,idx_n,(void *)dx_idx,(void *)dx_val,ALPHA_SPARSE_INDEXTYPE_I32,ALPHA_SPARSE_INDEX_BASE_ZERO,ALPHA_R_64F);

    alphasparseDnVecDescr_t y{};
    alphasparseCreateDnVec(&y,idx_n * 20,(void *)dy,ALPHA_R_64F);

    // Call cudasparse csrmv
    alphasparseAxpby(handle, (void *)&alpha, x, (void *)&beta, y),

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(alpha_y, dy, sizeof(double) * idx_n * 20, cudaMemcpyDeviceToHost);

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    alphasparse_destory_handle(handle);
}

static void cpu_axpby()
{
    for (int i = 0; i < idx_n * 20; i ++) {
        cpu_y[i] = beta * cpu_y[i];
    }
    for (int i = 0; i < idx_n; i ++) {
        cpu_y[alpha_x_idx[i]] = alpha * x_val[i] + cpu_y[alpha_x_idx[i]];
    }
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file  = args_get_data_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    iter  = args_get_iter(argc, argv);
    idx_n  = args_get_nnz(argc, argv);
    alpha_x_idx =
        (int *)alpha_memalign(sizeof(int) * idx_n, DEFAULT_ALIGNMENT);
    cuda_x_idx = (int *)alpha_memalign(sizeof(int) * idx_n,
                                                DEFAULT_ALIGNMENT);
    x_val     = (double *)alpha_memalign(sizeof(double) * idx_n, DEFAULT_ALIGNMENT);
    alpha_y   = (double *)alpha_memalign(sizeof(double) * idx_n * 20, DEFAULT_ALIGNMENT);
    cuda_y     = (double *)alpha_memalign(sizeof(double) * idx_n * 20, DEFAULT_ALIGNMENT);
    cpu_y     = (double *)alpha_memalign(sizeof(double) * idx_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random(alpha_y, 1, idx_n * 20);
    alpha_fill_random(cuda_y, 1, idx_n * 20);
    alpha_fill_random(cpu_y, 1, idx_n * 20);
    alpha_fill_random(x_val, 0, idx_n);

    for (int i = 0; i < idx_n; i++) {
        alpha_x_idx[i] = i * 20;
        cuda_x_idx[i]   = i * 20;
    }

    alpha_axpby();

    if (check_flag) {
        cuda_axpby();
        cpu_axpby();
        check(alpha_y, idx_n * 20, cuda_y, idx_n * 20);
        check(cpu_y, idx_n * 20, cuda_y, idx_n * 20);
        check(cpu_y, idx_n * 20, alpha_y, idx_n * 20);
    }
    printf("\n");

    alpha_free(x_val);
    alpha_free(cuda_x_idx);
    alpha_free(alpha_x_idx);
    return 0;
}
