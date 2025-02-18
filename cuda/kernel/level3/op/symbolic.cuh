#ifndef __Z_SYMBOLIC_CUH__
#define __Z_SYMBOLIC_CUH__

#include "define.h"

// good pwarp
template <typename IndexType>
__global__ void __launch_bounds__(PWARP_BLOCK_SIZE, 2) k_symbolic_shared_hash_pwarp(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol,
    int * __restrict__ d_bins, 
    int bin_size,
    IndexType * __restrict__ d_row_nnz){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x & (PWARP - 1);
    int rid = i / PWARP;
    int block_rid = rid & (PWARP_ROWS - 1);

    __shared__ IndexType shared_mem[PWARP_ROWS * PWARP_TSIZE + PWARP_ROWS];
    int *shared_table = (int *)shared_mem;
    int *shared_nnz = (int *)(shared_mem + PWARP_ROWS * PWARP_TSIZE);
    int j, k;
    for(j = threadIdx.x; j < PWARP_ROWS * PWARP_TSIZE; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x < PWARP_ROWS){
        shared_nnz[threadIdx.x] = 0;
    }
    if(rid >= bin_size){
        return;
    }
    __syncthreads();
    int *table = shared_table + block_rid * PWARP_TSIZE;

    rid = d_bins[rid];
    int acol, bcol;
    int hash, old;
    for(j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP){ // pwarp per row, thread per a item, thread per b row
        acol = d_acol[j];
        for(k = d_brpt[acol]; k < d_brpt[acol + 1]; k++){ // thread per b row
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) & (PWARP_TSIZE - 1);
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(table + hash, -1, bcol);
                if(old == -1){
                    atomicAdd(shared_nnz + block_rid, 1);
                    break;
                }
                else if(old == bcol){
                    break;
                }
                else{
                    hash = (hash + 1) & (PWARP_TSIZE - 1);
                }
#endif
#ifdef HASH_MULTI
                if(table[hash] == bcol){
                    break;
                }
                else if (table[hash] == -1){
                    old = atomicCAS(table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz + block_rid, 1);
                        break;
                    }
                }
                else{
                    hash = (hash + 1) &(PWARP_TSIZE - 1);
                }
#endif
            }
        }
    }
    __syncthreads();
    if(tid == 0){
        d_row_nnz[rid] = shared_nnz[block_rid];
    }
}

template <typename IndexType, int SH_ROW>
__global__ void __launch_bounds__(1024, 2) k_symbolic_shared_hash_tb(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol,
    int * __restrict__ d_bins,
    IndexType * __restrict__ d_row_nnz){

    //long long t0 = clock64();

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ int shared_table[SH_ROW];
    __shared__ IndexType shared_nnz[1];

    for(j = threadIdx.x; j < SH_ROW; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
        //shared_nnz[0] = threadIdx.x  + *d_fail_bin_size;
    }
    __syncthreads();
    int acol, bcol, hash, old;
    int rid = d_bins[blockIdx.x];
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) & (SH_ROW - 1);
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_table + hash, -1, bcol);
                if(old == -1){
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else if(old == bcol){
                    break;
                }
                else{
                    hash = (hash + 1) & (SH_ROW - 1);
                }
#endif
#ifdef HASH_MULTI
                if(shared_table[hash] == bcol){
                    break;
                }
                else if (shared_table[hash] == -1){
                    old = atomicCAS(shared_table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz, 1);
                        break;
                    }
                }
                else{
                    hash = (hash + 1) &(SH_ROW - 1);
                }
#endif
            }
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        d_row_nnz[rid] = shared_nnz[0];
    }

}

template <typename IndexType>
__global__ void __launch_bounds__(1024,2) k_symbolic_large_shared_hash_tb(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol,
    int * __restrict__ d_bins, 
    IndexType * __restrict__ d_row_nnz){

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ IndexType shared_mem[12288];
    const int tsize = 12287;
    int* shared_table = (int *)shared_mem;
    int* shared_nnz = (int *)(shared_mem + tsize);
    
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
    }
    __syncthreads();
    
    int rid = d_bins[blockIdx.x];
    int acol, bcol, hash, old;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_table + hash, -1, bcol);
                if(old == bcol){
                    break;
                }
                else if(old == -1){
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if(shared_table[hash] == bcol){
                    break;
                }
                else if (shared_table[hash] == -1){
                    old = atomicCAS(shared_table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz, 1);
                        break;
                    }
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }

#endif
            }
        }
    }
    __syncthreads();
    
    if(threadIdx.x == 0){
        d_row_nnz[rid] = shared_nnz[0];
    }
}

template <typename IndexType>
__global__ void __launch_bounds__(1024,1) k_symbolic_max_shared_hash_tb_with_fail(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol,
    int * __restrict__ d_bins, 
    int * __restrict__ d_fail_bins,
    int * __restrict__ d_fail_bin_size,
    IndexType * __restrict__ d_row_nnz){

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    extern __shared__ IndexType shared_mem[]; // size 24576
    const int tsize = 24575;
    int* shared_table = (int *)shared_mem;
    int* shared_nnz = (int *)(shared_mem + tsize);
    
    IndexType thresh_nnz = tsize * THRESH_SCALE;
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
    }
    __syncthreads();
    
    int rid = d_bins[blockIdx.x];
    int acol, bcol, hash, old;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(shared_nnz[0] <= thresh_nnz){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_table + hash, -1, bcol);
                if(old == bcol){
                    break;
                }
                else if(old == -1){
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if(shared_table[hash] == bcol){
                    break;
                }
                else if (shared_table[hash] == -1){
                    old = atomicCAS(shared_table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz, 1);
                        break;
                    }
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();

    int row_nnz;
    int fail_index;
    if(threadIdx.x == 0){
        row_nnz = shared_nnz[0];
        if(row_nnz <= thresh_nnz){ // success
            d_row_nnz[rid] = row_nnz;
        }
        else{ // fail case
            fail_index = atomicAdd(d_fail_bin_size, 1);
            d_fail_bins[fail_index] = rid;
        }
    }
}

template <typename IndexType>
__global__ void __launch_bounds__(1024, 2) k_symbolic_global_hash_tb(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol,
    int * __restrict__ d_bins,
    IndexType * __restrict__ d_row_nnz, 
    int * __restrict__ d_table,
    IndexType max_tsize){ 

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ IndexType shared_nnz[1];

    int rid = d_bins[blockIdx.x];
    int *table = d_table + blockIdx.x * max_tsize;
    int tsize = d_row_nnz[rid] * SYMBOLIC_SCALE_LARGE;
    int acol, bcol, hash, old;
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
    }
    __syncthreads();
    
    int nnz = 0;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(table + hash, -1, bcol);
                if(old == -1){
                    nnz++;
                    break;
                }
                else if(old == bcol){
                    break;
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if(table[hash] == bcol){
                    break;
                }
                else if (table[hash] == -1){
                    old = atomicCAS(table + hash, -1, bcol);
                    if(old == -1){
                        nnz++;
                        break;
                    }
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    atomicAdd(shared_nnz, nnz);

    __syncthreads();
    if(threadIdx.x == 0){
        d_row_nnz[rid] = shared_nnz[0];
    }
        
}


#endif
