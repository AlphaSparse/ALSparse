#pragma once

#include "alphasparse.h"
#include "alphasparse/types.h" 

template<typename T, typename U>
__global__ static void
spsv_csr_u_up_cw_kernel(
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const U* csr_val,
    const T m,
    const U alpha,
    const U* x,
    volatile U* y,
    volatile T* get_value,
    T* id_extractor
) {
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    row_id = m - 1 - row_id;
    U tmp_sum = {};
    T ptr = csr_row_ptr[row_id + 1] - 1;
    T col_id;
    if (ptr >= 0) {
        col_id = csr_col_idx[ptr];;
    }
    while (ptr >= csr_row_ptr[row_id] - 1) {
        if (ptr == csr_row_ptr[row_id] - 1 || csr_col_idx[ptr] <= row_id) {
            y[row_id] = alpha * x[row_id] - tmp_sum;
            // Make sure that y[row_id] has been written before changing get_value[row_id]
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
        if (get_value[col_id] == 1) {
            // Get yi from global memory without "__threadfence()".
            tmp_sum += y[col_id] * csr_val[ptr];
            ptr--;
            if (ptr >= 0) {
                col_id = csr_col_idx[ptr]; 
            }
        }
    }
    return;
}


template<typename T>
__global__ static void
spsv_csr_u_up_cw_kernel(
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const cuFloatComplex* csr_val,
    const T m,
    const cuFloatComplex alpha,
    const cuFloatComplex* x,
    volatile cuFloatComplex* y,
    volatile T* get_value,
    T* id_extractor
) {
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    row_id = m - 1 - row_id;
    cuFloatComplex tmp_sum = {};
    T ptr = csr_row_ptr[row_id + 1] - 1;
    T col_id;
    if (ptr >= 0) {
        col_id = csr_col_idx[ptr];
    }
    while (ptr >= csr_row_ptr[row_id] - 1) {
        if (ptr == csr_row_ptr[row_id] - 1 || csr_col_idx[ptr] <= row_id) {
            cuFloatComplex yi = alpha * x[row_id] - tmp_sum;
            y[row_id].x = yi.x;
            y[row_id].y = yi.y;
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
        if (get_value[col_id] == 1) {
            cuFloatComplex yi;
            yi.x = y[col_id].x;
            yi.y = y[col_id].y;
            tmp_sum += yi * csr_val[ptr];
            ptr--;
            if (ptr >= 0) {
                col_id = csr_col_idx[ptr]; 
            }
        }
    }
    return;
}


template<typename T>
__global__ static void
spsv_csr_u_up_cw_kernel(
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const cuDoubleComplex* csr_val,
    const T m,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* x,
    volatile cuDoubleComplex* y,
    volatile T* get_value,
    T* id_extractor
) {
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    row_id = m - 1 - row_id;
    cuDoubleComplex tmp_sum = {};
    T ptr = csr_row_ptr[row_id + 1] - 1;
    T col_id;
    if (ptr >= 0) {
        col_id = csr_col_idx[ptr];
    }
    while (ptr >= csr_row_ptr[row_id] - 1) {
        if (ptr == csr_row_ptr[row_id] - 1 || csr_col_idx[ptr] <= row_id) {
            cuDoubleComplex yi = alpha * x[row_id] - tmp_sum;
            y[row_id].x = yi.x;
            y[row_id].y = yi.y;
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
        if (get_value[col_id] == 1) {
            cuDoubleComplex yi;
            yi.x = y[col_id].x;
            yi.y = y[col_id].y;
            tmp_sum += yi * csr_val[ptr];
            ptr--;
            if (ptr >= 0) {
                col_id = csr_col_idx[ptr]; 
            }
        }
    }
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_u_up_cw(
    alphasparseHandle_t handle,
    T m,
    T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_ind,
    const U* x,
    U* y,
    void *externalBuffer
) {
    const int threadPerBlock = 256;
    const int blockPerGrid = (m - 1) / threadPerBlock + 1;

    // get_value mem: sizeof(T) * m
    T *get_value = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(get_value, 0, m * sizeof(T));
    // id_extractor mem: sizeof(T) * 1
    T *id_extractor = reinterpret_cast<T*>(reinterpret_cast<char*>(get_value) + sizeof(T) * m);
    cudaMemset(id_extractor, 0, sizeof(T));

    spsv_csr_u_up_cw_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        csr_row_ptr,
        csr_col_ind,
        csr_val,
        m,
        alpha,
        x,
        y,
        get_value,
        id_extractor
    );

    return ALPHA_SPARSE_STATUS_SUCCESS;
}