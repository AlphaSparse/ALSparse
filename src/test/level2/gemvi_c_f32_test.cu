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
int *alpha_x_idx;
int *cuda_x_idx;
cuFloatComplex *x_val, *cuda_y, *alpha_y;
const cuFloatComplex alpha = {1.1f,2.4f};
const cuFloatComplex beta = {3.2f,4.3f};
alphasparseOperation_t transAT;

int m, n, nnz;
int *coo_row_index, *coo_col_index;
cuFloatComplex* coo_values;
int lda;
int sizeA;
cuFloatComplex * A;

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

static void cuda_gemvi() {
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    cudaDeviceProp devProp;
    int device_id = 0;

    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    int *dx_idx = NULL;
    cuFloatComplex *dx_val = NULL;
    cuFloatComplex *dy = NULL;
    cuFloatComplex *dA = NULL;

    cusparseOperation_t transA;
    if(transAT == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    else if(transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transA = CUSPARSE_OPERATION_TRANSPOSE;
    else transA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

    CHECK_CUDA(cudaMalloc((void **)&dx_idx, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc((void **)&dx_val, sizeof(cuFloatComplex) * n));
    CHECK_CUDA(cudaMalloc((void **)&dy, sizeof(cuFloatComplex) * m));
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(cuFloatComplex) * sizeA));

    CHECK_CUDA(cudaMemcpy(dx_idx, alpha_x_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx_val, x_val, sizeof(cuFloatComplex) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, cuda_y, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, A, sizeof(cuFloatComplex) * sizeA, cudaMemcpyHostToDevice));   

    int pBufferSize;
    void * pBuffer;
    cusparseCgemvi_bufferSize(handle, transA, m, n, nnz, &pBufferSize);
    CHECK_CUDA(cudaMalloc((void **)&pBuffer, sizeof(CUDA_R_32F) * pBufferSize));
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    CHECK_CUSPARSE(cusparseCgemvi(handle, transA, m, n, &alpha, dA, lda, nnz, dx_val, dx_idx, &beta, dy, base, pBuffer));

    // Device synchronization
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(cuda_y, dy, sizeof(cuFloatComplex) * m, cudaMemcpyDeviceToHost));

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    cudaFree(dA);
    cusparseDestroy(handle);
}

static void alpha_gemvi()
{
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    // Offload data to device
    int *dx_idx = NULL;
    cuFloatComplex *dx_val     = NULL;
    cuFloatComplex *dy         = NULL;
    cuFloatComplex *dA         = NULL;

    CHECK_CUDA(cudaMalloc((void **)&dx_idx, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc((void **)&dx_val, sizeof(cuFloatComplex) * n));
    CHECK_CUDA(cudaMalloc((void **)&dy, sizeof(cuFloatComplex) * m));
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(cuFloatComplex) * sizeA));

    CHECK_CUDA(cudaMemcpy(dx_idx, alpha_x_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx_val, x_val, sizeof(cuFloatComplex) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, alpha_y, sizeof(cuFloatComplex) * m, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, A, sizeof(cuFloatComplex) * sizeA, cudaMemcpyHostToDevice));

    int pBufferSize;
    void * pBuffer = NULL;
    cudaMalloc((void **)&pBuffer, sizeof(cuFloatComplex) );
    alphasparseIndexBase_t base = ALPHA_SPARSE_INDEX_BASE_ZERO;
    alphasparseCgemvi(handle, transAT, m, n, &alpha, dA, lda, nnz, dx_val, dx_idx, &beta, dy, base, pBuffer),

    // Device synchronization
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(alpha_y, dy, sizeof(cuFloatComplex) * m, cudaMemcpyDeviceToHost));

    // Clear up on device
    cudaFree(dx_val);
    cudaFree(dx_idx);
    cudaFree(dy);
    cudaFree(dA);
    // alphasparse_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file = args_get_data_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    transAT = alpha_args_get_transA(argc, argv);

    alpha_read_coo<cuFloatComplex>(
      file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);

    if(transAT != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
    {
      int t = n;
      n = m;
      m = t;
    }
    float spasity = (float)nnz / (m * n) ;
    nnz = m * spasity; 
    nnz = nnz > n ? n : nnz;
    lda = n;
    sizeA = m * lda;

    alpha_x_idx=(int *)alpha_memalign(sizeof(int) * nnz, DEFAULT_ALIGNMENT);
    cuda_x_idx = (int *)alpha_memalign(sizeof(int) * nnz,  DEFAULT_ALIGNMENT);

    x_val   = (cuFloatComplex *)alpha_memalign(sizeof(cuFloatComplex) * n, DEFAULT_ALIGNMENT);
    alpha_y = (cuFloatComplex *)alpha_memalign(sizeof(cuFloatComplex) * m, DEFAULT_ALIGNMENT);
    cuda_y  = (cuFloatComplex *)alpha_memalign(sizeof(cuFloatComplex) * m, DEFAULT_ALIGNMENT);
    A       = (cuFloatComplex *)alpha_memalign(sizeof(cuFloatComplex) * sizeA, DEFAULT_ALIGNMENT);

    alpha_fill_random(alpha_y, 1, m);
    memcpy(cuda_y, alpha_y, sizeof(cuFloatComplex)* m);
    alpha_fill_random(x_val, 0, n);
    alpha_fill_random(A, 1, sizeA);
    int fen = n / nnz;
    for (int i = 0; i < nnz; i++) {
        if(i == 0)
        {
          alpha_x_idx[i] = rand() % fen;
          cuda_x_idx[i]  = alpha_x_idx[i];
        }
        else
        {
          alpha_x_idx[i] = alpha_x_idx[i - 1] + rand() % fen;
          cuda_x_idx[i]  = alpha_x_idx[i];
        }
    }

    alpha_gemvi();

    if (check_flag) {
        cuda_gemvi();
        check(alpha_y, m, cuda_y, m);

        for(int i=0;i<10;i++)
            std::cout<<alpha_y[i]<<"  ";

        std::cout<<std::endl;
        for(int i=0;i<10;i++)
            std::cout<<cuda_y[i]<<"  ";   

        std::cout<<std::endl;   
    }
    printf("\n");

    alpha_free(x_val);
    alpha_free(cuda_x_idx);
    alpha_free(alpha_x_idx);
    return 0;
}