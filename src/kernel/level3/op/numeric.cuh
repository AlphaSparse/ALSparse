#ifndef __Z_NUMERIC_CUH__
#define __Z_NUMERIC_CUH__

#include "define.h"


// full occu
template <typename IndexType, typename DataType>
__global__ void __launch_bounds__(NUMERIC_PWARP_BLOCK_SIZE, 2) k_numeric_shared_hash_pwarp(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const DataType * __restrict__ d_aval,
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol, 
    const DataType * __restrict__ d_bval,
    int *d_bins, int bin_size,
    IndexType *d_crpt, IndexType *d_ccol, DataType* d_cval, const DataType alpha){

    //long long t0 = clock64();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x % NUMERIC_PWARP;
    int rid = i / NUMERIC_PWARP;
    int block_rid = rid % NUMERIC_PWARP_ROWS;
    __shared__ IndexType shared_mem[NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE * (sizeof(IndexType) + sizeof(DataType))/sizeof(IndexType)];
    const int tsize = NUMERIC_PWARP_TSIZE - 1;
    int *mono_shared_col = (int *)shared_mem;
    int *mono_shared_offset = (int *)(shared_mem + NUMERIC_PWARP_ROWS * tsize);
    DataType *mono_shared_val = (DataType*)(shared_mem + NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE);
    int j, k;
    for(j = threadIdx.x; j < NUMERIC_PWARP_ROWS * tsize; j += blockDim.x){
        mono_shared_col[j] = -1;
        mono_shared_val[j] = 0;
    }
    if(threadIdx.x < NUMERIC_PWARP_ROWS){
        mono_shared_offset[threadIdx.x] = 0;
    }
    if(rid >= bin_size){
        return;
    }
    __syncthreads();

    rid = d_bins[rid];
    int *shared_col = mono_shared_col + block_rid * tsize;
    //IndexType *shared_offset = shared_col + tsize;
    DataType *shared_val = mono_shared_val + block_rid * tsize;
    int acol, bcol, hash, old;
    DataType aval, bval;
    for(j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += NUMERIC_PWARP){ // pwarp per row, thread per a item, thread per b row
        acol = d_acol[j];
        aval = d_aval[j] * alpha;
        for(k = d_brpt[acol]; k < d_brpt[acol + 1]; k++){ // thread per b row
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (shared_col[hash] == bcol) {
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else if (shared_col[hash] == -1) {
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash pwarp hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    //__syncthreads();

    int c_offset = d_crpt[rid];
    int row_nnz = d_crpt[rid + 1] - d_crpt[rid];
    int offset;
    bool valid;
    #pragma unroll
    for(j = 0; j < tsize; j += NUMERIC_PWARP){
        offset = j + tid;
        valid = offset < tsize;
        if(valid){
            acol = shared_col[offset];
            aval = shared_val[offset];
            if(acol != -1){
                offset = atomicAdd(mono_shared_offset + block_rid, 1);
            }
        }
        __syncthreads();
        if(valid && acol != -1){
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }

    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash pwarp condense %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // count sort the result
    for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
        acol = shared_col[j];
        offset = 0;
        for (k = 0; k < row_nnz; k++) {
            offset += (unsigned int)(shared_col[k] - acol) >> 31;
        }
        d_ccol[c_offset + offset] = shared_col[j];
        d_cval[c_offset + offset] = shared_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}
}

template <typename IndexType, typename DataType, int SH_ROW, int BS>
__global__ void __launch_bounds__(1024,2) k_numeric_shared_hash_tb_full_occu(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const DataType * __restrict__ d_aval,
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol, 
    const DataType * __restrict__ d_bval,
    int *d_bins,
    IndexType *d_crpt, IndexType *d_ccol, DataType* d_cval, const DataType alpha){

    //long long t0 = clock64();

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ IndexType shared_mem[SH_ROW * (sizeof(IndexType) + sizeof(DataType))/sizeof(IndexType)];
    const int tsize = SH_ROW - 1;
    int *shared_col = (int *)shared_mem;
    int *shared_offset = (int *)(shared_mem + (SH_ROW - 1));
    DataType* shared_val = (DataType*)(shared_mem + SH_ROW);

    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_col[j] = -1;
        shared_val[j] = 0;
    }
    if(threadIdx.x == 0){
        shared_offset[0] = 0;
    }
    __syncthreads();

    int acol, bcol, hash, old;
    DataType aval, bval;
    int rid = d_bins[blockIdx.x];
    int c_offset = d_crpt[rid];
    int row_nnz = d_crpt[rid + 1] - d_crpt[rid];

    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        aval = d_aval[j] * alpha;
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (shared_col[hash] == bcol) {
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else if (shared_col[hash] == -1) {
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }

    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // condense shared hash table
    int offset;
    bool valid;
    #pragma unroll
    for (j = 0; j < SH_ROW; j += BS){
        offset = j + threadIdx.x;
        valid = offset < tsize;
        if(valid){
            acol = shared_col[offset];
            aval = shared_val[offset];
            if(acol != -1){
                offset = atomicAdd(shared_offset, 1);
            }
        }
        __syncthreads();
        if(valid && acol != -1){
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }
    
    // count sort the result
    __syncthreads();
    int count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        target = shared_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++) {
            count += (unsigned int)(shared_col[k] - target) >> 31;
        }
        d_ccol[c_offset + count] = shared_col[j];
        d_cval[c_offset + count] = shared_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}

}


// half occu due to max shared memory
template <typename IndexType, typename DataType>
__global__ void __launch_bounds__(1024, 1) k_numeric_max_shared_hash_tb_half_occu(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const DataType * __restrict__ d_aval,
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol, 
    const DataType * __restrict__ d_bval,
    int *d_bins,
    IndexType *d_crpt, IndexType *d_ccol, DataType* d_cval, const DataType alpha){

    //long long t0 = clock64();

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    extern __shared__ IndexType shared_mem[];
    const int tsize = 8191;
    int *shared_col = (int *)shared_mem;
    int *shared_offset = (int *)(shared_mem + tsize);
    DataType* shared_val = (DataType*)(shared_mem + (tsize + 1));

    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_col[j] = -1;
        shared_val[j] = 0;
    }
    if(threadIdx.x == 0){
        shared_offset[0] = 0;
    }
    __syncthreads();

    int acol, bcol, hash, old;
    DataType aval, bval;
    int rid = d_bins[blockIdx.x];
    int c_offset = d_crpt[rid];
    int row_nnz = d_crpt[rid + 1] - d_crpt[rid];

    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        aval = d_aval[j] * alpha;
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (shared_col[hash] == bcol) {
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else if (shared_col[hash] == -1) {
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // condense shared hash table
    int offset;
    bool valid;
    #pragma unroll
    for (j = 0; j < 8192; j += 1024){
        offset = j + threadIdx.x;
        valid = offset < tsize;
        if(valid){
            acol = shared_col[offset];
            aval = shared_val[offset];
            if(acol != -1){
                offset = atomicAdd(shared_offset, 1);
            }
        }
        __syncthreads();
        if(valid && acol != -1){
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }

    // count sort the result
    __syncthreads();
    int count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        target = shared_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++) {
            count += (unsigned int)(shared_col[k] - target) >> 31;
        }
        d_ccol[c_offset + count] = shared_col[j];
        d_cval[c_offset + count] = shared_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}
}

template <typename IndexType, typename DataType>
__global__ void __launch_bounds__(1024, 2) k_numeric_global_hash_tb_full_occu(
    const IndexType * __restrict__ d_arpt, const IndexType * __restrict__ d_acol, 
    const DataType * __restrict__ d_aval,
    const IndexType * __restrict__ d_brpt, const IndexType * __restrict__ d_bcol, 
    const DataType * __restrict__ d_bval,
    int *d_bins, IndexType max_tsize, int* d_tables,
    IndexType *d_crpt, IndexType *d_ccol, DataType* d_cval, const DataType alpha){

    //long long t0 = clock64();

    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ IndexType shared_offset[1];
    
    int* table_col = d_tables + blockIdx.x * max_tsize * ((sizeof(IndexType) + sizeof(DataType))/sizeof(IndexType));
    DataType* table_val = (DataType*)(table_col + max_tsize);
    int rid = d_bins[blockIdx.x];
    int c_offset = d_crpt[rid];
    int row_nnz = d_crpt[rid + 1] - c_offset;
    int tsize = row_nnz * NUMERIC_SCALE_LARGE;
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        table_col[j] = -1;
        table_val[j] = 0;
    }
    if(threadIdx.x == 0){
        shared_offset[0] = 0;
    }
    __syncthreads();

    int acol, bcol, hash, old;
    DataType aval, bval;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        aval = d_aval[j] * alpha;
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(table_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(table_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (table_col[hash] == bcol) {
                    atomicAdd(table_val + hash, aval * bval);
                    break;
                }
                else if (table_col[hash] == -1) {
                    old = atomicCAS(table_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(table_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }

#endif
            }
        }
    }

    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash global hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // condense shared hash table
    __syncthreads();
    int offset;
    for (j = threadIdx.x; j < tsize; j += blockDim.x){
        acol = table_col[j];
        aval = table_val[j] * alpha;
        if(acol != -1){
            offset = atomicAdd(shared_offset, 1);
            d_ccol[c_offset + offset] = acol;
            d_cval[c_offset + offset] = aval;
        }
    }
    __syncthreads();
    for(j = threadIdx.x; j < row_nnz; j += blockDim.x){
        table_col[j] = d_ccol[c_offset + j];
        table_val[j] = d_cval[c_offset + j];
    }

    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash global  condense %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // count sort the result
    __syncthreads();
    int count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        target = table_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++) {
            count += (unsigned int)(table_col[k] - target) >> 31;
        }
        d_ccol[c_offset + count] = table_col[j];
        d_cval[c_offset + count] = table_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}
}


#endif
