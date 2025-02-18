#pragma once

#include "alphasparse.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

template <typename T>
__global__ static void
coo2csr_kernel(const T* row_data, T nnz, T m, T* csrRowPtr)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = tid; i < nnz; i += stride)
    {
        atomicAdd(&csrRowPtr[row_data[i]], 1);
    }
    if(tid == 0) csrRowPtr[m] = nnz;
}

template <typename T>
__global__ static void
zero_kernel(T m, T* csrRowPtr)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = tid; i < max_align_t; i += stride)
    {
        csrRowPtr[i] = 0;
    }
}