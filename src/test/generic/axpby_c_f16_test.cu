
#include "../test_common.h"

/**
 * @brief ict mv csr test
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

#include "alphasparse/util/auxiliary.h"

const char *file;
bool check_flag;
int iter;

// sparse vector
int nnz;
int *alpha_x_idx;
int *roc_x_idx;
half2 *x_val, *cuda_y, *alpha_y;
cuFloatComplex alpha = make_cuFloatComplex(1.,1.);
cuFloatComplex beta = make_cuFloatComplex(1.,1.);

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

static void roc_axpby() {
    cudaDeviceProp devProp;
    int device_id = 0;

    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    int *dx_idx = NULL;
    half2 *dx_val = NULL;
    half2 *dy = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * idx_n);
    cudaMalloc((void **)&dx_val, sizeof(half2) * idx_n);
    cudaMalloc((void **)&dy, sizeof(half2) * idx_n * 20);

    cudaMemcpy(dx_idx, roc_x_idx, sizeof(int) * idx_n,
            cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, x_val, sizeof(half2) * idx_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, cuda_y, sizeof(half2) * idx_n * 20, cudaMemcpyHostToDevice);

    // rocSPARSE handle
    cusparseHandle_t     handle = NULL;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    cusparseSpVecDescr_t x;
    cusparseCreateSpVec(&x, idx_n * 20, idx_n, (void *)dx_idx,
                                (void *)dx_val, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_C_16F);

    cusparseDnVecDescr_t y;
    cusparseCreateDnVec(&y, idx_n * 20, (void *)dy,
                                CUDA_C_16F);

    // Call rocsparse csrmv
    roc_call_exit(cusparseAxpby(handle, (void *)&alpha, x, (void *)&beta, y),
                "rocsparse_axpby");

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_y, dy, sizeof(half2) * idx_n * 20, cudaMemcpyDeviceToHost);

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    cusparseDestroy(handle);
}

static void alpha_axpyi()
{
    // rocSPARSE handle
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
    half2 *dx_val     = NULL;
    half2 *dy         = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * idx_n);
    cudaMalloc((void **)&dx_val, sizeof(half2) * idx_n);
    cudaMalloc((void **)&dy, sizeof(half2) * idx_n * 20);

    cudaMemcpy(dx_idx, alpha_x_idx, sizeof(int) * idx_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, x_val, sizeof(half2) * idx_n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, alpha_y, sizeof(half2) * idx_n * 20, cudaMemcpyHostToDevice);

    alphasparseSpVecDescr_t x{};
    alphasparseCreateSpVec(&x,idx_n * 20,idx_n,(void *)dx_idx,(void *)dx_val,ALPHA_SPARSE_INDEXTYPE_I32,ALPHA_SPARSE_INDEX_BASE_ZERO,ALPHA_C_16F);

    alphasparseDnVecDescr_t y{};
    alphasparseCreateDnVec(&y,idx_n * 20,(void *)dy,ALPHA_C_16F);

    // Call rocsparse csrmv
    alphasparseAxpby(handle, (void *)&alpha, x, (void *)&beta, y),

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(alpha_y, dy, sizeof(half2) * idx_n * 20, cudaMemcpyDeviceToHost);

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
    idx_n  = args_get_nnz(argc, argv);

    alpha_x_idx =
        (int *)alpha_memalign(sizeof(int) * idx_n, DEFAULT_ALIGNMENT);
    roc_x_idx = (int *)alpha_memalign(sizeof(int) * idx_n,
                                                DEFAULT_ALIGNMENT);
    x_val     = (half2 *)alpha_memalign(sizeof(half2) * idx_n, DEFAULT_ALIGNMENT);
    alpha_y   = (half2 *)alpha_memalign(sizeof(half2) * idx_n * 20, DEFAULT_ALIGNMENT);
    cuda_y     = (half2 *)alpha_memalign(sizeof(half2) * idx_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random(alpha_y, 1, idx_n * 20);
    alpha_fill_random(cuda_y, 1, idx_n * 20);
    alpha_fill_random(x_val, 0, idx_n);

    for (int i = 0; i < idx_n; i++) {
        alpha_x_idx[i] = i * 20;
        roc_x_idx[i]   = i * 20;
    }
    printf("\n====icty raw========\n");
    for(int i=0;i<10;i++)
        std::cout<<alpha_y[i]<<std::endl;
    printf("\n====alpha_y raw end========\n");
    for(int i=0;i<10;i++)
        std::cout<<cuda_y[i]<<std::endl;
    printf("\n====cuda_y raw========\n");

    alpha_axpyi();

    if (check_flag) {
        roc_axpby();
        printf("\n====icty raw========\n");
        for(int i=0;i<10;i++)
            std::cout<<alpha_y[i]<<std::endl;
        printf("\n====alpha_y raw end========\n");
        for(int i=0;i<10;i++)
            std::cout<<cuda_y[i]<<std::endl;
        printf("\n====cuda_y raw========\n");
        check(alpha_y, idx_n * 20, cuda_y, idx_n * 20);
    }
    printf("\n");

    alpha_free(x_val);
    alpha_free(roc_x_idx);
    alpha_free(alpha_x_idx);
    return 0;
}
