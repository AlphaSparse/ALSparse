#pragma once

#include "alphasparse.h"
#include "alphasparse/types.h" 

// alg_num: 2

// Compute vector y for a lower left matrix in CSR format.
// One thread processes one matrix row.
template<typename T, typename U>
__global__ static void
spsv_csr_n_lo_cw_kernel(
    const T* csr_row_ptr,   /* <I> */
    const T* csr_col_idx,   /* <I> */
    const U* csr_val,       /* <I> */
    const T m,              /* <I> */
    const U alpha,          /* <I> */
    const U* x,             /* <I> */
    U* y,                   /* <O> */
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
    T col_id = 0;
    U tmp_sum = {};
    // Process the specific row from matrix left to diagonal.
    // ptr indicates the index of csr_val corresponding to the row.
    T ptr = csr_row_ptr[row_id];
    while (ptr < csr_row_ptr[row_id + 1]) {
        // Get current NNZ's 2D location - column number.
        col_id = csr_col_idx[ptr];
        while (get_value[col_id] == 1) {
            // Always update tmp_sum when the current thread finds that the NNZ 
            // it encounters has been written.
            __threadfence();
            // Get tmp_sum from global memory.
            tmp_sum = tmp_sum + y[col_id] * csr_val[ptr];
            ptr++;
            col_id = csr_col_idx[ptr];
        }
        if (col_id == row_id) {
            // When the thread handle the diagonal NNZ, it updates yi and refreshes global memory.
            y[row_id] = (alpha * x[row_id] - tmp_sum) / csr_val[ptr];
            __threadfence();
            // When yi has been written to global memory, the get_value signal needs to change to 1.
            get_value[row_id] = 1;
            return;
        }
    }
}


// Similar to above kernel.
// There is one difference that define get_value volatile.
// double, float
template<typename T, typename U>
// typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, float>::value, T>::type 
__global__ static void
spsv_csr_n_lo_cw_kernel_volatile(
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
    T ptr = csr_row_ptr[row_id];
    U tmp_sum = {};
    // T col_id;
    T col_id = csr_col_idx[ptr];
    while (ptr < csr_row_ptr[row_id + 1]) {
        // col_id = csr_col_idx[ptr];
        while (get_value[col_id] == 1) {
            // Get yi from global memory without "__threadfence()".
            tmp_sum = tmp_sum + y[col_id] * csr_val[ptr];
            // tmp_sum = tmp_sum + (__builtin_nontemporal_load(&y[col_id])) * csr_val[ptr]; 编译器不支持？
            ptr++;
            col_id = csr_col_idx[ptr];
        }
        if (col_id == row_id) {
            y[row_id] = (alpha * x[row_id] - tmp_sum) / csr_val[ptr];
            // (__builtin_nontemporal_store(&y[row_id])) = (alpha * x[row_id] - tmp_sum) / csr_val[ptr];
            // Make sure that y[row_id] has been written before changing get_value[row_id]
            __threadfence();    
            get_value[row_id] = 1;
            return;
        }
    }
    return;
}

// cuDoubleComplex
template<typename T>
__global__ static void
spsv_csr_n_lo_cw_kernel_volatile(
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
    T ptr = csr_row_ptr[row_id];
    cuDoubleComplex tmp_sum = {};
    T col_id = csr_col_idx[ptr];
    while (ptr < csr_row_ptr[row_id + 1]) {
        while (get_value[col_id] == 1) {
            cuDoubleComplex yi;
            yi.x = y[col_id].x;
            yi.y = y[col_id].y;
            tmp_sum = tmp_sum + yi * csr_val[ptr];
            ptr++;
            col_id = csr_col_idx[ptr];
        }
        if (col_id == row_id) {
            cuDoubleComplex yi = (alpha * x[row_id] - tmp_sum) / csr_val[ptr];
            y[row_id].x = yi.x;
            y[row_id].y = yi.y;
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
    }
}

// cuFloatComplex
template<typename T>
__global__ static void
spsv_csr_n_lo_cw_kernel_volatile(
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
    T ptr = csr_row_ptr[row_id];
    cuFloatComplex tmp_sum = {};
    T col_id = csr_col_idx[ptr];
    while (ptr < csr_row_ptr[row_id + 1]) {
        while (get_value[col_id] == 1) {
            cuFloatComplex yi;
            yi.x = y[col_id].x;
            yi.y = y[col_id].y;
            tmp_sum = tmp_sum + yi * csr_val[ptr];
            ptr++;
            col_id = csr_col_idx[ptr];
        }
        if (col_id == row_id) {
            cuFloatComplex yi = (alpha * x[row_id] - tmp_sum) / csr_val[ptr];
            y[row_id].x = yi.x;
            y[row_id].y = yi.y;
            __threadfence();
            get_value[row_id] = 1;
            return;
        }
    }
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_cw(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
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

    // spsv_csr_n_lo_cw_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
    //   csr_row_ptr,
    //   csr_col_ind,
    //   csr_val,
    //   m,
    //   alpha,
    //   x,
    //   y,
    //   get_value,
    //   id_extractor
    // );

    spsv_csr_n_lo_cw_kernel_volatile<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
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
