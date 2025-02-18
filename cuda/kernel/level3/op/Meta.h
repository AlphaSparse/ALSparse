#ifndef __Z_META__
#define __Z_META__

#include "define.h"
#include "alphasparse.h"
#include "alphasparse/types.h" 

class Meta{
    public:
    // first, allocate C.rpt. 
    // d_row_flop, d_estimated_row_nnz, d_row_nnz are all reused with C.rpt

    // combined memory
    int *d_combined_mem; // second, allocate for all others
    int *combined_mem; // second, allocate for all others

    // meta data
    int M; // number of rows
    int N; // number of cols
    int *d_bins; // size M
    int *d_bin_size; // size NUM_BIN
    int *d_bin_offset; // size NUM_BIN
    int *d_max_row_nnz; // size 1
    int *d_total_nnz; // size 1
    int *d_cub_storage; // size variable
    int *bin_size; // size NUM_BIN
    int *bin_offset; // size NUM_BIN
    int *max_row_nnz; // size 1
    int *total_nnz; // size 1
    size_t cub_storage_size;
    cudaStream_t *stream;


    // symbolic global and numeric global, is allocated at runtime
    int* d_global_mem_pool; // size unknown, allocated at runtime
    size_t global_mem_pool_size;
    bool global_mem_pool_malloced;

    // ********************************************************
    // public method
    Meta(){}
    Meta(const Meta&) = delete;
    Meta &operator=(const Meta&) = delete;
    Meta(alphasparseSpMatDescr_t& C); // init and first malloc
    void allocate_rpt(alphasparseSpMatDescr_t& C);
    void allocate(alphasparseSpMatDescr_t& C); // malloc conbined mem and pin the variables
    void release();

    void memset_bin_size(int stream_idx); // set d_bin_size only to 0
    void memset_all(int stream_idx); // set d_bin_size and other to 0
    void D2H_bin_size(int stream_idx);
    void D2H_all(int stream_idx);
    void H2D_bin_offset(int stream_idx);
    ~Meta();
};

#endif
