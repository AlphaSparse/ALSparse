#pragma once

#include "alphasparse/types.h"
#include "alphasparse/util/simplelist.h"
#include "alphasparse/spdef.h"

// TODO use somplelist to record optimizing history instead of a bool
// TODO remove alpha,beta from inspector
typedef enum
{
    ALPHA_ATOMIC,
    ALPHA_PRIVATE
} alpha_conflict_solution_t;

typedef enum
{
    ALPHA_BALANCE_DIRECT = 0, // For instance, ELL gets task by directly computing row index range
    ALPHA_BALANCE_NNZ, // NNZ numbers matters 
    ALPHA_BALANCE_HYB, // undefined
} alpha_load_balance_method_t;

typedef struct alphasparse_inspector_mv_
{
    ALPHA_INT optimized_record_id; // encode following info into a decimal number
    double *matrix_features;     // features for gemv format selection
    ALPHA_INT feature_len;         //feature length
    ALPHA_INT num_threads;         // thread resources for real computation
    ALPHA_INT iter;                // expected call times of mv kernels

    struct alpha_matrix_descr descr_in;    // matrix description of requested kernel
    alphasparseOperation_t operation_in; // matrix operation of requested kernel
    
    // following are output : indeed they shall be called Selector
    alphasparse_matrix_t optimal_matrix_base; // optimal matrix handle for computation
    struct alpha_matrix_descr descr_out;       // optimal matrix description of requested kernel
    alphasparseOperation_t operation_out;    // optimal matrix operation of requested kernel

    alpha_conflict_solution_t conflict_solution; // for kernels that have wirte conflicts
    alpha_load_balance_method_t partition_method; // defaulted to ALPHA_BALANCE_DIRECT
    // ALPHA_Number alpha; // kernel selection
    // ALPHA_Number beta;  // kernel selection

} alphasparse_inspector_mv;
typedef struct alphasparse_inspector_sv_
{
    double *matrix_features; // features for gemv format selection
    ALPHA_INT feature_len;     //feature length
    ALPHA_INT num_threads;     // thread resources for real computation
    ALPHA_INT iter;            // expected call times of kernels

    struct alpha_matrix_descr descr_in;    // matrix description of requested kernel
    alphasparseOperation_t operation_in; // matrix operation of requested kernel

    // following are output : indeed they shall be called Selector
    alphasparse_matrix_t optimal_matrix_base; // optimal matrix handle for computation
    struct alpha_matrix_descr descr_out;       // optimal matrix description of requested kernel
    alphasparseOperation_t operation_out;    // optimal matrix operation of requested kernel
} alphasparse_inspector_sv;

typedef struct alphasparse_inspector_mmd_
{
    double *matrix_features;
    ALPHA_INT feature_len; //feature length
    ALPHA_INT num_threads; // thread resources for real computation
    ALPHA_INT iter;        // expected call times of kernels

    ALPHA_INT dense_columns;               // columns of rhs matrix
    alphasparse_layout_t dense_layout;    // layout of rhs matrix
    struct alpha_matrix_descr descr_in;    // lrs matrix description of requested kernel
    alphasparseOperation_t operation_in; // lrs matrix operation of requested kernel

    // following are output : indeed they shall be called Selector
    alphasparse_matrix_t optimal_matrix_base;   // optimal lrs matrix handle for computation
    alphasparseOperation_t operation_out;      // optimal lrs matrix operation of requested kernel
    struct alpha_matrix_descr descr_out;         // optimal lrs matrix description of requested kernel
    ALPHA_INT OPTIMAL_MTILE;                    // best mm tiling parameters learned
    ALPHA_INT OPTIMAL_KTILE;                    // best mm tiling parameters learned
    ALPHA_INT OPTIMAL_NTILE;                    // best mm tiling parameters learned
    alpha_conflict_solution_t conflict_solution; // for kernels that have wirte conflicts
} alphasparse_inspector_mmd;

typedef struct alphasparse_inspector_sm_
{
    double *matrix_features;
    ALPHA_INT feature_len; //feature length
    ALPHA_INT num_threads; // thread resources for real computation
    ALPHA_INT iter;        // expected call times of kernels

    ALPHA_INT dense_columns;               // columns of rhs matrix
    alphasparse_layout_t dense_layout;    // layout of rhs matrix
    struct alpha_matrix_descr descr_in;    // lrs matrix description of requested kernel
    alphasparseOperation_t operation_in; // lrs matrix operation of requested kernel

    //following are output : indeed they shall be called Selector
    alphasparse_matrix_t optimal_matrix_base;   // optimal lrs matrix handle for computation
    alphasparseOperation_t operation_out;      // optimal lrs matrix operation of requested kernel
    struct alpha_matrix_descr descr_out;         // optimal lrs matrix description of requested kernel
    ALPHA_INT OPTIMAL_MTILE;                    // best mm tiling parameters learned
    ALPHA_INT OPTIMAL_KTILE;                    // best mm tiling parameters learned
    ALPHA_INT OPTIMAL_NTILE;                    // best mm tiling parameters learned
    alpha_conflict_solution_t conflict_solution; // for kernels that have wirte conflicts
} alphasparse_inspector_sm;

typedef struct alphasparse_inspector_mm_
{
    ALPHA_INT optimized_record_id;
    double *matrix_features_lhs;
    double *matrix_features_rhs;
    ALPHA_INT feature_len;
    ALPHA_INT num_threads;
    ALPHA_INT iter;

    struct alpha_matrix_descr descr_lhs_in;
    alphasparseOperation_t operation_lhs_in;
    struct alpha_matrix_descr descr_rhs_in;
    alphasparseOperation_t operation_rhs_in;

    alphasparse_matrix_t optimal_matrix_lhs;
    struct alpha_matrix_descr descr_out_lhs;
    alphasparseOperation_t operation_out_lhs;

    alphasparse_matrix_t optimal_matrix_rhs;
    struct alpha_matrix_descr descr_out_rhs;
    alphasparseOperation_t operation_out_rhs;

    alpha_conflict_solution_t conflict_solution;

} alphasparse_inspector_mm;

typedef alphasparse_inspector_mv *alphasparse_inspector_mv_t;
typedef alphasparse_inspector_sv *alphasparse_inspector_sv_t;
typedef alphasparse_inspector_mmd *alphasparse_inspector_mmd_t;
typedef alphasparse_inspector_mm *alphasparse_inspector_mm_t;
typedef alphasparse_inspector_sm *alphasparse_inspector_sm_t;

typedef enum
{
    ALPHA_NONE = 0,
    ALPHA_MV,
    ALPHA_SV,
    ALPHA_MMD,
    ALPHA_MM, // MM is outlined since it takes two sparse matrices as input, alphasparse_optimize api doesn't fit
    ALPHA_SM
} alpha_kernel_t;

// typedef unsigned short request_bitmap_t; //indicate which kernels are requested. Lower 5 bits are used
//TODO request_kernel: simplelist
typedef struct alphasparse_inspector_
{
    alpha_kernel_t request_kernel;
    alphasparse_inspector_mv_t mv_inspector;
    alphasparse_inspector_sv_t sv_inspector;
    alphasparse_inspector_mmd_t mmd_inspector;
    alphasparse_inspector_mm_t mm_inspector;
    alphasparse_inspector_sm_t sm_inspector;

    alphasparse_memory_usage_t memory_policy;
} alphasparse_inspector;
