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
int nnz = 1000;
int *alpha_x_idx;
int *roc_x_idx;
nv_bfloat16 *alpha_x_val, *roc_x_val;
nv_bfloat16 *y;

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            exit(-1);                                                  \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            exit(-1);                                                      \
        }                                                                  \
    }

static void roc_gthr()
{
    // cusparse handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    cudaDeviceProp devProp;
    int device_id = 0;

    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    int *dx_idx = NULL;
    nv_bfloat16 *dx_val = NULL;
    nv_bfloat16 *dy = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * nnz);
    cudaMalloc((void **)&dx_val, sizeof(nv_bfloat16) * nnz);
    cudaMalloc((void **)&dy, sizeof(nv_bfloat16) * nnz * 20);

    cudaMemcpy(dx_idx, roc_x_idx, sizeof(int) * nnz,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, roc_x_val, sizeof(nv_bfloat16) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(nv_bfloat16) * nnz * 20, cudaMemcpyHostToDevice);

    cusparseSpVecDescr_t x;
    cusparseCreateSpVec(&x, nnz * 20, nnz, (void *)dx_idx,
                        (void *)dx_val, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_16BF);

    cusparseDnVecDescr_t y;
    cusparseCreateDnVec(&y, nnz * 20, (void *)dy,
                        CUDA_R_16BF);

    // Call cusparse csrmv
    CHECK_CUSPARSE(cusparseGather(handle, y, x))

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(roc_x_val, dx_val, sizeof(nv_bfloat16) * nnz, cudaMemcpyDeviceToHost);

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    cusparseDestroy(handle);
}

static void alpha_gthr()
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
    nv_bfloat16 *dx_val = NULL;
    nv_bfloat16 *dy = NULL;

    cudaMalloc((void **)&dx_idx, sizeof(int) * nnz);
    cudaMalloc((void **)&dx_val, sizeof(nv_bfloat16) * nnz);
    cudaMalloc((void **)&dy, sizeof(nv_bfloat16) * nnz * 20);

    cudaMemcpy(dx_idx, roc_x_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dx_val, alpha_x_val, sizeof(nv_bfloat16) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(nv_bfloat16) * nnz * 20, cudaMemcpyHostToDevice);

    alphasparseSpVecDescr_t x{};
    alphasparseCreateSpVec(&x, nnz * 20, nnz, (void *)dx_idx, (void *)dx_val,
                           ALPHA_SPARSE_INDEXTYPE_I32, ALPHA_SPARSE_INDEX_BASE_ZERO, ALPHA_R_16BF);

    alphasparseDnVecDescr_t y{};
    alphasparseCreateDnVec(&y, nnz * 20, (void *)dy, ALPHA_R_16BF);

    // Call cusparse csrmv
    alphasparseGather(handle, y, x);

    // Device synchronization
    cudaDeviceSynchronize();

    cudaMemcpy(alpha_x_val, dx_val, sizeof(nv_bfloat16) * nnz, cudaMemcpyDeviceToHost);

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
    file = args_get_data_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    iter = args_get_iter(argc, argv);
    nnz  = args_get_nnz(argc, argv);

    alpha_x_idx = (int *)alpha_memalign(sizeof(int) * nnz, DEFAULT_ALIGNMENT);
    roc_x_idx = (int *)alpha_memalign(sizeof(int) * nnz,
                                      DEFAULT_ALIGNMENT);
    alpha_x_val = (nv_bfloat16 *)alpha_memalign(sizeof(nv_bfloat16) * nnz, DEFAULT_ALIGNMENT);
    roc_x_val = (nv_bfloat16 *)alpha_memalign(sizeof(nv_bfloat16) * nnz, DEFAULT_ALIGNMENT);
    y = (nv_bfloat16 *)alpha_memalign(sizeof(nv_bfloat16) * nnz * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random(y, 0, nnz * 20);
    alpha_fill_random(alpha_x_val, 1, nnz);
    alpha_fill_random(roc_x_val, 1, nnz);

    for (int i = 0; i < nnz; i++)
    {
        alpha_x_idx[i] = i * 20;
        roc_x_idx[i] = i * 20;
    }

    alpha_gthr();

    if (check_flag)
    {
        roc_gthr();
        check(alpha_x_val, nnz, roc_x_val, nnz);
    }
    printf("\n");

    alpha_free(roc_x_val);
    alpha_free(alpha_x_val);
    alpha_free(roc_x_idx);
    alpha_free(alpha_x_idx);
    alpha_free(y);
    return 0;
}
