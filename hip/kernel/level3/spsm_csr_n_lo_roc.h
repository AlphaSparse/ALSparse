#pragma once

#include "alphasparse.h"
#include "../level2/spsv_csr_n_lo_roc.h"

template<typename T, typename U>
alphasparseStatus_t
spsm_csr_n_lo_roc_analysis(
    alphasparseHandle_t handle,
    const T m,              /* <I> A_rows | A_cols | B_rows | C_rows. */
    const T n_rhs,          /* <I> B_cols | C_cols. */
    const T nnz,            /* <I> */
    const U alpha,          /* <I> */
    const U *csr_val,       /* <I> */
    const T *csr_row_ptr,   /* <I> */
    const T *csr_col_idx,   /* <I> */
    T *row_map,
    const U *B,                   /* <I> */
    const T ldb,                  /* <I> */
    U *C,                    /* <O> */
    void *externalBuffer
) {
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 64;
    const dim3 threadPerBlock = dim3(BLOCKSIZE);
    const dim3 blockPerGrid = dim3((m - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
    T *done_array = reinterpret_cast<T*>(externalBuffer);
    hipMemset(done_array, 0, m * sizeof(T));
    spsv_csr_n_lo_roc_analysis_kernel<BLOCKSIZE, WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        m,
        csr_row_ptr,
        csr_col_idx,
        done_array,
        row_map
    );
    get_row_map_sorted(handle, m, done_array, row_map);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U>
alphasparseStatus_t
spsm_csr_n_lo_roc_solve(
    alphasparseHandle_t handle,
    const T m,              /* <I> A_rows | A_cols | B_rows | C_rows. */
    const T n_rhs,          /* <I> B_cols | C_cols. */
    const T nnz,            /* <I> */
    const U alpha,          /* <I> */
    const U *csr_val,       /* <I> */
    const T *csr_row_ptr,   /* <I> */
    const T *csr_col_idx,   /* <I> */
    U *B,                   /* <I> */
    T ldb,                  /* <I> */
    U *C                    /* <O> */
) {

    return ALPHA_SPARSE_STATUS_SUCCESS;
}