#ifndef CSRSPGEMM_DEVICE_OP
#define CSRSPGEMM_DEVICE_OP

#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "op/Meta.h"
#include "op/define.h"
#include <cuda_profiler_api.h>
#include "op/setup.cuh"
#include "op/binning.cuh"
#include "op/symbolic.cuh"
#include "op/numeric.cuh"

#include <cub/cub.cuh>

Meta::Meta(alphasparseSpMatDescr_t& C){
    allocate_rpt(C);
}

void Meta::allocate_rpt(alphasparseSpMatDescr_t& C){
    if(C->row_data == nullptr)
        CHECK_CUDA(cudaMalloc(&C->row_data, (C->rows + 1)*sizeof(int64_t)));
}

void Meta::allocate(alphasparseSpMatDescr_t& C){
    M = C->rows;
    N = C->cols;
    stream = new cudaStream_t [NUM_BIN];
    for(int i = 0; i < NUM_BIN; i++){
        CHECK_CUDA(cudaStreamCreate(stream + i));
    }
        
    cub::DeviceScan::ExclusiveSum(nullptr, cub_storage_size, (int *)C->row_data, (int *)C->row_data, M + 1); // calculate tmp_storage_size in bytes

    int d_combined_size = M  + 2 * NUM_BIN + 2 + cub_storage_size/(sizeof(int));
    CHECK_CUDA(cudaMalloc(&d_combined_mem, d_combined_size * sizeof(int)));
    int combined_size = 2 * NUM_BIN + 2;
    combined_mem = (int *)malloc(combined_size * sizeof(int));
    assert(combined_mem != nullptr);

    d_bins = (int *)d_combined_mem; // size M
    d_bin_size = (int *)d_combined_mem + M; // size NUM_BIN
    d_max_row_nnz = d_bin_size + NUM_BIN; // size 1
    d_total_nnz = d_bin_size + NUM_BIN + 1; // size 1
    d_bin_offset = d_total_nnz + 1; // size NUM_BIN
    d_cub_storage = d_bin_offset + 1;

    bin_size = (int*) combined_mem; // size NUM_BIN
    max_row_nnz = bin_size + NUM_BIN; // size 1
    total_nnz = bin_size + NUM_BIN + 1; // size 1
    bin_offset = bin_size + NUM_BIN + 2; // size NUM_BIN
    
    d_global_mem_pool = nullptr;
    global_mem_pool_size = 0;
    global_mem_pool_malloced = false;
}

void Meta::release(){
    cudaFree(d_combined_mem);
    d_combined_mem = nullptr;
    if(stream != nullptr){
        for(int i = 0; i < NUM_BIN; i++){
            cudaStreamDestroy(stream[i]);
        }
        delete [] stream;
        stream = nullptr;
    }
    delete [] combined_mem;
    combined_mem = nullptr;
}

Meta::~Meta(){
    release();
}


void Meta::memset_all(int stream_idx = 1){
    CHECK_CUDA(cudaMemsetAsync(d_bin_size, 0, (NUM_BIN + 2) * sizeof(int), stream[stream_idx]));
    //CHECK_CUDA(cudaMemset(d_bin_size, 0, (NUM_BIN + 5) * sizeof(int)));
}
void Meta::memset_bin_size(int stream_idx = 1){
    CHECK_CUDA(cudaMemsetAsync(d_bin_size, 0, NUM_BIN * sizeof(int), stream[stream_idx]));
    //CHECK_CUDA(cudaMemset(d_bin_size, 0, (NUM_BIN + 5) * sizeof(int)));
}

void Meta::D2H_all(int stream_idx = 0){
    CHECK_CUDA(cudaMemcpyAsync(bin_size, d_bin_size, (NUM_BIN + 2) * sizeof(int), cudaMemcpyDeviceToHost, stream[stream_idx]));
    //CHECK_CUDA(cudaMemcpy(bin_size, d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice));
}

void Meta::D2H_bin_size(int stream_idx = 0){
    CHECK_CUDA(cudaMemcpyAsync(bin_size, d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost, stream[stream_idx]));
    //CHECK_CUDA(cudaMemcpy(bin_size, d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice));
}

void Meta::H2D_bin_offset(int stream_idx = 0){
    CHECK_CUDA(cudaMemcpyAsync(d_bin_offset, bin_offset, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice, stream[stream_idx]));
}

void cudaruntime_warmup(){
    int *d;
    CHECK_CUDA(cudaMalloc(&d, 4));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename IndexType>
void h_compute_flop(const alphasparseSpMatDescr_t& A, const alphasparseSpMatDescr_t& B, alphasparseSpMatDescr_t& C, Meta& meta){
    int BS = 1024;
    int GS = div_up(A->rows, BS);
    k_compute_flop<<<GS, BS>>>((IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType)C->rows,(IndexType *)C->row_data, (IndexType *)(C->row_data + C->rows));
}

// setup
template <typename IndexType>
void h_setup(const alphasparseSpMatDescr_t& A, const alphasparseSpMatDescr_t& B, alphasparseSpMatDescr_t& C, Meta& meta){
    meta.allocate_rpt(C); // allocate (IndexType *)C->row_data, other init procedure, default stream
    cudaMemset((IndexType *)C->row_data + (IndexType)C->rows, 0, sizeof(IndexType));
    h_compute_flop<IndexType>(A, B, C, meta); // compute flop, stream[0]
    meta.allocate(C); // allocate other memory    
    CHECK_CUDA(cudaMemcpy(meta.max_row_nnz, (IndexType *)C->row_data + (IndexType)C->rows, sizeof(IndexType), cudaMemcpyDeviceToHost));
}

// symbolic binning
template <typename IndexType>
inline void h_symbolic_binning(alphasparseSpMatDescr_t &C, Meta& meta){
    meta.memset_all(0); // memset d_bin_size
    int BS = 1024;
    int GS = div_up((IndexType)C->rows, BS);
    if(*meta.max_row_nnz <= 26){
        k_binning_small<<<GS, BS>>>(meta.d_bins, (IndexType)C->rows);
        meta.bin_size[0] = (IndexType)C->rows;
        for(int i = 1; i< NUM_BIN; i++){
            meta.bin_size[i] = 0;
        }
        meta.bin_offset[0] = 0;
        for(int i = 1; i < NUM_BIN; i++){
            meta.bin_offset[i] = (IndexType)C->rows;
        }

    }
    else{
        k_symbolic_binning<<<GS, BS, 0, meta.stream[0]>>>(
            (IndexType *)C->row_data, (IndexType)C->rows, meta.d_bin_size);
        meta.D2H_bin_size(0);
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for(int i = 0; i < NUM_BIN - 1; i++){
            meta.bin_offset[i+1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.H2D_bin_offset(0);
        k_symbolic_binning2<<<GS, BS, 0, meta.stream[0]>>>(
            (IndexType *)C->row_data, (IndexType)C->rows, 
            meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
    }
}

template <typename IndexType>
void h_symbolic(const alphasparseSpMatDescr_t& A, const alphasparseSpMatDescr_t& B, alphasparseSpMatDescr_t& C, Meta& meta){
    //double t0, t1;
    if(meta.bin_size[5]){
        k_symbolic_shared_hash_tb<IndexType, 8192><<<meta.bin_size[5], 1024, 0, meta.stream[5]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[5],
            (IndexType *)C->row_data);
    }
    int *d_fail_bins, *d_fail_bin_size;
    int fail_bin_size = 0;
    if(meta.bin_size[7]){ // shared hash with fail
        //t0 = fast_clock_time();
        if(meta.bin_size[7] + 1 <= meta.cub_storage_size/sizeof(int)){
            d_fail_bins = meta.d_cub_storage;
            d_fail_bin_size = meta.d_cub_storage + meta.bin_size[7];
        }
        else{ // allocate global memory
            CHECK_CUDA(cudaMalloc(&d_fail_bins, (meta.bin_size[7] + 1) * sizeof(int)));
            d_fail_bin_size = d_fail_bins + meta.bin_size[7];
        }
        CHECK_CUDA(cudaMemsetAsync(d_fail_bin_size, 0, sizeof(int), meta.stream[7]));
        CHECK_CUDA(cudaFuncSetAttribute(k_symbolic_max_shared_hash_tb_with_fail<IndexType>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_symbolic_max_shared_hash_tb_with_fail<IndexType>
            <<<meta.bin_size[7], 1024, 98304, meta.stream[7]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[7],
            d_fail_bins, d_fail_bin_size,
            (IndexType *)C->row_data);

    }
    if(meta.bin_size[6]){
        k_symbolic_large_shared_hash_tb<<<meta.bin_size[6], 1024, 0, meta.stream[6]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[6],
            (IndexType *)C->row_data);
    }
    if(meta.bin_size[0]){
        int BS = PWARP_ROWS * PWARP;
        int GS = div_up(meta.bin_size[0], PWARP_ROWS);
        k_symbolic_shared_hash_pwarp<<<GS, BS, 0, meta.stream[0]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[0],
            meta.bin_size[0],
            (IndexType *)C->row_data);
    }

    if(meta.bin_size[7]){
    CHECK_CUDA(cudaMemcpyAsync(&fail_bin_size, d_fail_bin_size, sizeof(int), cudaMemcpyDeviceToHost, meta.stream[7]));
    CHECK_CUDA(cudaStreamSynchronize(meta.stream[7]));
        if(fail_bin_size){ // global hash
            //printf("inside h_symbolic fail_bin_size %d\n", fail_bin_size);
            IndexType max_tsize = *meta.max_row_nnz * SYMBOLIC_SCALE_LARGE;
            meta.global_mem_pool_size = fail_bin_size * max_tsize * sizeof(int);
            CHECK_CUDA(cudaMalloc(&meta.d_global_mem_pool, meta.global_mem_pool_size));
            meta.global_mem_pool_malloced = true;
            k_symbolic_global_hash_tb<<<fail_bin_size, 1024, 0, meta.stream[7]>>>(
                (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
                d_fail_bins,
                (IndexType *)C->row_data, meta.d_global_mem_pool, max_tsize);
        }
    }
    

    if(meta.bin_size[4]){
        k_symbolic_shared_hash_tb<IndexType, 4096><<<meta.bin_size[4], 512, 0, meta.stream[4]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[4],
            (IndexType *)C->row_data);
    }


    if(meta.bin_size[3]){
        k_symbolic_shared_hash_tb<IndexType, 2048><<<meta.bin_size[3], 256, 0, meta.stream[3]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[3],
            (IndexType *)C->row_data);
    }
    if(meta.bin_size[2]){
        k_symbolic_shared_hash_tb<IndexType, 1024><<<meta.bin_size[2], 128, 0, meta.stream[2]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[2],
            (IndexType *)C->row_data);
    }
    if(meta.bin_size[1]){
        k_symbolic_shared_hash_tb<IndexType, 512><<<meta.bin_size[1], 64, 0, meta.stream[1]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (IndexType *)B->row_data, (IndexType *)B->col_data,
            meta.d_bins + meta.bin_offset[1],
            (IndexType *)C->row_data);
    }


    if(meta.bin_size[7] && meta.bin_size[7] + 1 > meta.cub_storage_size/sizeof(int)){
        CHECK_CUDA(cudaFree(d_fail_bins));
    }
}

template <typename IndexType>
inline void h_numeric_binning(alphasparseSpMatDescr_t C, Meta& meta){
    meta.memset_all(0);
    int BS = 1024;
    int GS = div_up((IndexType)C->rows, BS);
    k_numeric_binning<IndexType><<<GS, BS, 0 , meta.stream[0]>>>((IndexType *)C->row_data, (IndexType)C->rows,
        meta.d_bin_size, meta.d_total_nnz, meta.d_max_row_nnz);
    meta.D2H_all(0);
    CHECK_CUDA(cudaStreamSynchronize(meta.stream[0]));
    if(*meta.max_row_nnz <= 16){
        k_binning_small<IndexType><<<GS, BS>>>(meta.d_bins, (IndexType)C->rows);
        meta.bin_size[0] = (IndexType)C->rows;
        for(int i = 1; i< NUM_BIN; i++){
            meta.bin_size[i] = 0;
        }
        meta.bin_offset[0] = 0;
        for(int i = 1; i < NUM_BIN; i++){
            meta.bin_offset[i] = (IndexType)C->rows;
        }
    }
    else{
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for(int i = 0; i < NUM_BIN - 1; i++){
            meta.bin_offset[i+1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.H2D_bin_offset(0);

        k_numeric_binning2<IndexType><<<GS, BS, 0, meta.stream[0]>>>((IndexType *)C->row_data, (IndexType)C->rows,
            meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
    }
}


template <typename IndexType, typename DataType>
inline void h_numeric_full_occu(const alphasparseSpMatDescr_t& A, const alphasparseSpMatDescr_t& B, alphasparseSpMatDescr_t C, Meta& meta, const DataType alpha)
{
    if(meta.bin_size[6]){
        CHECK_CUDA(cudaFuncSetAttribute(k_numeric_max_shared_hash_tb_half_occu<IndexType, DataType>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_numeric_max_shared_hash_tb_half_occu<IndexType, DataType><<<meta.bin_size[6], 1024, 98304, meta.stream[6]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[6],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }


    if(meta.bin_size[7]){ // global bin
        //printf("inside h_numeric_phase max_row_nnz %d\n", *meta.max_row_nnz);
        IndexType max_tsize = *meta.max_row_nnz * NUMERIC_SCALE_LARGE;
        size_t global_size = meta.bin_size[7] * max_tsize * (sizeof(IndexType) + sizeof(DataType));
        if(meta.global_mem_pool_malloced){
            if(global_size <= meta.global_mem_pool_size){
                // do nothing
            }
            else{
                CHECK_CUDA(cudaFree(meta.d_global_mem_pool));
                CHECK_CUDA(cudaMalloc(&meta.d_global_mem_pool, global_size));
            }
        }
        else{
            CHECK_CUDA(cudaMalloc(&meta.d_global_mem_pool, global_size));
            meta.global_mem_pool_size = global_size;
            meta.global_mem_pool_malloced = true;
        }
        k_numeric_global_hash_tb_full_occu<<<meta.bin_size[7], 1024, 0, meta.stream[7]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[7], max_tsize, meta.d_global_mem_pool,
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }

    if(meta.bin_size[5]){
        k_numeric_shared_hash_tb_full_occu<IndexType, DataType, 4096, 1024>
            <<<meta.bin_size[5], 1024, 0, meta.stream[5]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[5],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }
    if(meta.bin_size[0]){
        int BS = NUMERIC_PWARP_ROWS * NUMERIC_PWARP;
        int GS = div_up(meta.bin_size[0], NUMERIC_PWARP_ROWS);
        k_numeric_shared_hash_pwarp<<<GS, BS, 0, meta.stream[0]>>>(
            (IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[0], meta.bin_size[0],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }

    if(meta.bin_size[4]){
        k_numeric_shared_hash_tb_full_occu<IndexType, DataType, 2048, 512>
            <<<meta.bin_size[4], 512, 0, meta.stream[4]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[4],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }
    if(meta.bin_size[3]){
        k_numeric_shared_hash_tb_full_occu<IndexType, DataType, 1024, 256>
            <<<meta.bin_size[3], 256, 0, meta.stream[3]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[3],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }

    if(meta.bin_size[2]){
        k_numeric_shared_hash_tb_full_occu<IndexType, DataType, 512, 128>
            <<<meta.bin_size[2], 128, 0, meta.stream[2]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[2],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }
    if(meta.bin_size[1]){
        k_numeric_shared_hash_tb_full_occu<IndexType, DataType, 256, 64>
            <<<meta.bin_size[1], 64, 0, meta.stream[1]>>>
            ((IndexType *)A->row_data, (IndexType *)A->col_data, (DataType *)A->val_data, (IndexType *)B->row_data, (IndexType *)B->col_data, (DataType *)B->val_data,
            meta.d_bins + meta.bin_offset[1],
            (IndexType *)C->row_data, (IndexType *)C->col_data, (DataType *)C->val_data, alpha);
    }

    if(meta.global_mem_pool_malloced){
        CHECK_CUDA(cudaFree(meta.d_global_mem_pool));
    }
}

#endif