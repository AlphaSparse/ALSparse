#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"

// Compute vector y for a unit-diag lower left matrix in CSR format.
// One thread processes one matrix row.
// double, float
template<typename T, typename U>
__global__ static void
spsv_csr_u_lo_cw_kernel(
    const T* csr_row_ptr,   /* <I> */
    const T* csr_col_idx,   /* <I> */
    const U* csr_val,       /* <I> */
    const T m,              /* <I> */
    const U alpha,          /* <I> */
    const U* x,             /* <I> */
    volatile U* y,          /* <O> */
    volatile T* get_value,  /* <I> Signals indicate whether components have been written. 
                                Length: m.
                                * Initially, the get_value value of each row is 0, the value of specific row will be
                                changed to 1 when the yi of the row has been computed and updated.
                                * Volatile valiable definition allows read/write variables to 
                                access global memory directly, without be cached. 
                                Make sure the consistency of threads access to memory. */
    T* id_extractor         /* <I> Indicates the index a thread processes.
                                Length: 1
                                * Initially, the value is 0. */
) {
    // Threads get sequential processing indices by the time they are put into operation.
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    U tmp_sum = {};
    // Process the specific row from matrix left to diagonal.
    // ptr indicates the index of csr_val corresponding to the row.
    T ptr = csr_row_ptr[row_id];
    T col_id = csr_col_idx[ptr];    // NNZ's 2D location - column number.
    bool flag = true;
    while (flag && ptr <= csr_row_ptr[row_id + 1]) {
        // If the NNZs to the left of a diaonal line have been scanned,
		// update corresponding y element, and then the thread finish its task.
		if (ptr == csr_row_ptr[row_id + 1] || csr_col_idx[ptr] >= row_id) {
			y[row_id] = alpha * x[row_id] - tmp_sum;
			__threadfence();
			get_value[row_id] = 1;
			// return;
			// break;
			flag = false;
		}
		// If current corresponding y element has been written to global memory,
        // update tmp_sum.
        if (get_value[col_id] == 1) {
            tmp_sum += y[col_id] * csr_val[ptr];
            ptr++;
            col_id = csr_col_idx[ptr];
        }
    }
    return;
}

// hipFloatComplex
template<typename T>
__global__ static void
spsv_csr_u_lo_cw_kernel(
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
    hipFloatComplex tmp_sum = {};
    T ptr = csr_row_ptr[row_id];
    T col_id = csr_col_idx[ptr];
    while (ptr <= csr_row_ptr[row_id + 1]) {
        if (ptr == csr_row_ptr[row_id + 1] || csr_col_idx[ptr] >= row_id) {
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
            tmp_sum = tmp_sum + yi * csr_val[ptr];
            ptr++;
            col_id = csr_col_idx[ptr];
        }
    }
    return;
}

// hipDoubleComplex
template<typename T>
__global__ static void
spsv_csr_u_lo_cw_kernel(
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
    hipDoubleComplex tmp_sum = {};
    T ptr = csr_row_ptr[row_id];
    T col_id = csr_col_idx[ptr];
    while (ptr <= csr_row_ptr[row_id + 1]) {
        if (ptr == csr_row_ptr[row_id + 1] || csr_col_idx[ptr] >= row_id) {
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
            ptr++;
            col_id = csr_col_idx[ptr];
        }
    }
    return;
}


template<typename T, typename U>
alphasparseStatus_t
spsv_csr_u_lo_cw(
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

    spsv_csr_u_lo_cw_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
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
