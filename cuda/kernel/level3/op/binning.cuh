#ifndef __Z_ESTIMATE_SYMBOLIC_BINNING_CUH__
#define __Z_ESTIMATE_SYMBOLIC_BINNING_CUH__

#include "define.h"


template <typename IndexType>
__global__ void __launch_bounds__(1024, 2) k_symbolic_binning(
    IndexType *d_row_flop, IndexType M, int *d_bin_size){
    
    __shared__ IndexType shared_bin_size[NUM_BIN];
    if(threadIdx.x < NUM_BIN){
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
    IndexType row_nnz, j;
    //IndexType range[NUM_BIN] = {32, 512, 1024, 2048,     4096, 8192, 12287, INT_MAX}; // 1x
    IndexType range[NUM_BIN] = {26, 426, 853, 1706,     3413, 6826, 10240, INT_MAX}; // 1.2x
    //IndexType range[NUM_BIN] = {21, 341, 682, 1365,     2730, 5461, 8191, INT_MAX}; // 1.5x
    if(i < M){
        row_nnz = d_row_flop[i];
        //#pragma unroll
        for(j = 0; j < NUM_BIN; j++){
            if(row_nnz <= range[j]){
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
    before_end:
    __syncthreads();
    if(threadIdx.x < NUM_BIN){
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
}

template <typename IndexType>
__global__ void __launch_bounds__ (1024, 2) k_symbolic_binning2(
    IndexType * __restrict__ d_row_flop, 
    IndexType M, 
    int * __restrict__ d_bins, 
    int * __restrict__ d_bin_size, 
    int * __restrict__ d_bin_offset){


    __shared__ IndexType shared_bin_size[NUM_BIN];
    __shared__ IndexType shared_bin_offset[NUM_BIN];
    if(threadIdx.x < NUM_BIN){
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
    IndexType row_nnz, j;
    //IndexType range[NUM_BIN] = {32, 512, 1024, 2048,     4096, 8192, 12287, INT_MAX}; // 1x
    IndexType range[NUM_BIN] = {26, 426, 853, 1706,     3413, 6826, 10240, INT_MAX}; // 1.2x
    //IndexType range[NUM_BIN] = {21, 341, 682, 1365,     2730, 5461, 8191, INT_MAX}; // 1.5x
    if(i < M){
        row_nnz = d_row_flop[i];
        //#pragma unroll
        for(j = 0; j < NUM_BIN; j++){
            if(row_nnz <= range[j]){
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
    before_end:

    __syncthreads();
    if(threadIdx.x < NUM_BIN){
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    IndexType index;
    if(i < M){
        //#pragma unroll
        for(j = 0; j < NUM_BIN; j++){
            if(row_nnz <= range[j]){
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
    }
}

template <typename IndexType>
__global__ void k_binning_small(
    int *d_bins, IndexType M){

    IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= M){
        return;
    }
    d_bins[i] = i;
}


template <typename IndexType>
__global__ void __launch_bounds__ (1024, 2) k_numeric_binning(
    IndexType * __restrict__ d_row_nnz, 
    IndexType M, 
    int * __restrict__ d_bin_size, 
    int * __restrict__ d_total_nnz, 
    int * __restrict__ d_max_row_nnz){

    __shared__ IndexType shared_bin_size[NUM_BIN];
    __shared__ IndexType shared_local_nnz[1];
    __shared__ IndexType shared_max_row_nnz[1];
    if(threadIdx.x < NUM_BIN){
        shared_bin_size[threadIdx.x] = 0;
    }
    if(threadIdx.x == 32){
        shared_local_nnz[0] = 0;
        shared_max_row_nnz[0] = 0;
    }
    __syncthreads();
    //IndexType range[NUM_BIN] = {31, 255, 511, 1022,    2047, 4095, 8191, INT_MAX}; // 1x
    //IndexType range[NUM_BIN] = {21, 192, 384, 768,    1536, 3072, 5460, INT_MAX}; // 1.5x
    IndexType range[NUM_BIN] = {16, 128, 256, 512,    1024, 2048, 4095, INT_MAX}; // 2x
    //IndexType range[NUM_BIN] = {10, 85, 170, 341,    682, 1365, 2730, INT_MAX}; // 3x
    IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
    IndexType row_nnz, j;
    if(i < M){
        row_nnz = d_row_nnz[i];
        atomicAdd(shared_local_nnz, row_nnz);
        atomicMax(shared_max_row_nnz, row_nnz);
        //#pragma unroll
        for(j = 0; j < NUM_BIN; j++){
            if(row_nnz <= range[j]){
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
    before_end:


    __syncthreads();
    if(threadIdx.x < NUM_BIN){
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
    if(threadIdx.x == 32){
        atomicAdd(d_total_nnz, shared_local_nnz[0]);
    }
    if(threadIdx.x == 64){
        atomicMax(d_max_row_nnz, shared_max_row_nnz[0]);
    }
}

template <typename IndexType>
__global__ void  __launch_bounds__ (1024, 2) k_numeric_binning2 (
    IndexType * __restrict__ d_row_nnz, 
    IndexType M, 
    int * __restrict__ d_bins, 
    int * __restrict__ d_bin_size, 
    int * __restrict__ d_bin_offset){ 

    __shared__ IndexType shared_bin_size[NUM_BIN];
    __shared__ IndexType shared_bin_offset[NUM_BIN];
    if(threadIdx.x < NUM_BIN){
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    //IndexType range[NUM_BIN] = {31, 255, 511, 1022,    2047, 4095, 8191, INT_MAX}; // 1x
    //IndexType range[NUM_BIN] = {21, 192, 384, 768,    1536, 3072, 5460, INT_MAX}; // 1.5x
    IndexType range[NUM_BIN] = {16, 128, 256, 512,    1024, 2048, 4095, INT_MAX}; // 2x
    //IndexType range[NUM_BIN] = {10, 85, 170, 341,    682, 1365, 2730, INT_MAX}; // 3x
    IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
    IndexType row_nnz, j;
    if(i < M){
        row_nnz = d_row_nnz[i];
        //#pragma unroll
        for(j = 0; j < NUM_BIN; j++){
            if(row_nnz <= range[j]){
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
    before_end:


    __syncthreads();
    if(threadIdx.x < NUM_BIN){
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    IndexType index;
    if(i < M){
        //#pragma unroll
        for(j = 0; j < NUM_BIN; j++){
            if(row_nnz <= range[j]){
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
    }
}

#endif
