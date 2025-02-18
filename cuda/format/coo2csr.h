#pragma once

#include "alphasparse.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime_api.h>

__device__ static int lower_bound(const int* arr, int key, int low, int high)
{
    while(low < high)
    {
        int mid = low + ((high - low) >> 1);

        if(arr[mid] < key)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }

    return low;
}

template <typename T>
__global__ static void
coo2csr_kernel(const int* row_data, int nnz, int m, T* csrRowPtr)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_base = 0;
    if(gid >= m)
    {
        return;
    }

    if(gid == 0)
    {
        csrRowPtr[0] = idx_base;
        csrRowPtr[m] = nnz + idx_base;
        return;
    }
    
    csrRowPtr[gid] = lower_bound(row_data, gid + idx_base, static_cast<int>(0), nnz) + idx_base;
}

template <typename T>
__global__ static void
coo2csr_kernel_1(const int* row_data, int nnz, int m, T* csrRowPtr)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_base = 0;    

    for(int i = gid; i < nnz; i+= stride)
    {
        atomicAdd(&csrRowPtr[row_data[i]], 1);
    }   

    if(gid == 0)
    {
        csrRowPtr[m] = nnz + idx_base;
    } 
}

template <typename T>
__global__ static void
zero_kernel(const int* row_data, int nnz, int m, T* csrRowPtr)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_base = 0;

    for(int i = gid; i < m; i+= stride)
    {
        csrRowPtr[i] = idx_base;
    }
}

template <typename T>
alphasparseStatus_t
alphasparseXcoo2csr(const int* row_data, int nnz, int m, T* csrRowPtr)
{
    if(csrRowPtr == nullptr) return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    printf("converting COO to CSR\n");
    zero_kernel<<<64,256>>>(row_data, nnz, m, csrRowPtr);
    coo2csr_kernel_1<T><<<64,256>>>(row_data, nnz, m, csrRowPtr);
    thrust::exclusive_scan(thrust::device, csrRowPtr, csrRowPtr + m, csrRowPtr);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
