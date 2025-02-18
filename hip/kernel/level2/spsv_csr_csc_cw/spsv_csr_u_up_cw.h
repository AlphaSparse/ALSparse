#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"

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
	bool flag = true;
    while (flag && ptr >= csr_row_ptr[row_id] - 1) {
        if (ptr == csr_row_ptr[row_id] - 1 || csr_col_idx[ptr] <= row_id) {
            y[row_id] = alpha * x[row_id] - tmp_sum;
            // Make sure that y[row_id] has been written before changing get_value[row_id]
            __threadfence();
            get_value[row_id] = 1;
            flag = false;
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
    const hipFloatComplex* csr_val,
    const T m,
    const hipFloatComplex alpha,
    const hipFloatComplex* x,
    volatile hipFloatComplex* y,
    volatile T* get_value,
    T* id_extractor
) {
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    row_id = m - 1 - row_id;
    hipFloatComplex tmp_sum = {};
    T ptr = csr_row_ptr[row_id + 1] - 1;
    T col_id;
    if (ptr >= 0) {
        col_id = csr_col_idx[ptr];
    }
    while (ptr >= csr_row_ptr[row_id] - 1) {
        if (ptr == csr_row_ptr[row_id] - 1 || csr_col_idx[ptr] <= row_id) {
            hipFloatComplex yi = alpha * x[row_id] - tmp_sum;
            y[row_id].x = yi.x;
            y[row_id].y = yi.y;
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
        if (get_value[col_id] == 1) {
            hipFloatComplex yi;
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
    const hipDoubleComplex* csr_val,
    const T m,
    const hipDoubleComplex alpha,
    const hipDoubleComplex* x,
    volatile hipDoubleComplex* y,
    volatile T* get_value,
    T* id_extractor
) {
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    row_id = m - 1 - row_id;
    hipDoubleComplex tmp_sum = {};
    T ptr = csr_row_ptr[row_id + 1] - 1;
    T col_id;
    if (ptr >= 0) {
        col_id = csr_col_idx[ptr];
    }
    while (ptr >= csr_row_ptr[row_id] - 1) {
        if (ptr == csr_row_ptr[row_id] - 1 || csr_col_idx[ptr] <= row_id) {
            hipDoubleComplex yi = alpha * x[row_id] - tmp_sum;
            y[row_id].x = yi.x;
            y[row_id].y = yi.y;
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
        if (get_value[col_id] == 1) {
            hipDoubleComplex yi;
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
    hipMemset(get_value, 0, m * sizeof(T));
    // id_extractor mem: sizeof(T) * 1
    T *id_extractor = reinterpret_cast<T*>(reinterpret_cast<char*>(get_value) + sizeof(T) * m);
    hipMemset(id_extractor, 0, sizeof(T));

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
