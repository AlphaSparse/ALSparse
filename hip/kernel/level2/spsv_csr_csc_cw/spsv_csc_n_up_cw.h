#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"

// Preprocessing phase.
// Calculate the dependence of each row for an upper right matrix in CSC format.
// One thread processes one matrix column.
template<typename T>
__global__ static void
spsv_csc_n_up_cw_preprocess(
    const T* csc_col_ptr,   /* <I> */
    const T* csc_row_idx,   /* <I> */
    const T m,              /* <I> */
    T* in_degree            /* <O> Indicates the number of elements/NNZ that each y/component depends on.
                                Length: m. */
) {
    const T col_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_id >= m) {
        return;
    }
    // Add up the number of NNZs of encounter row for specific column from matrix top to diagnal.
    for (T ptr = csc_col_ptr[col_id]; ptr < csc_col_ptr[col_id + 1] && csc_row_idx[ptr] <= col_id; ptr++) {
        atomicAdd(&in_degree[csc_row_idx[ptr]], 1);   
    }
    return;
}

// Compute vector y for an upper right matrix in CSC format.
// One thread processes one matrix column.
template<typename T, typename U>
__global__ static void
spsv_csc_n_up_cw_kernel(
    const T* csc_col_ptr,   /* <I> */ 
    const T* csc_row_idx,   /* <I> */ 
    const U* csc_val,       /* <I> */ 
    const T m,              /* <I> */ 
    const U alpha,          /* <I> */ 
    const U* x,             /* <I> */ 
    U* y,                   /* <O> */ 
    T* id_extractor,        /* <I> Indicates the index a thread processes.
                                Length: 1
                                * Initially, the value is 0. */
    U* tmp_sum,             /* <I> Temporary cumulative value of each row.
                                Length: m. */
    T* in_degree            /* <I> The number of elements/NNZ that each y/component depends on, i.e. each yi's depedence.
                                Length: m. 
                                * Initially, the in_degree elements are calculated in the preprocessing phase, each 
                                element is greater than or equal to 1. */
) {
    // Threads get sequential processing indices by the time they are put into operation.
    T col_id = atomicAdd(id_extractor, 1);
    if (col_id >= m) {
        return;
    }
    col_id = m - 1 - col_id;
    T ptr = csc_col_ptr[col_id + 1] - 1;
    // Pass lower left area of the matrix.
    while (ptr > csc_col_ptr[col_id] && csc_row_idx[ptr] > col_id) {
        ptr--;
    }
    T row_id;
    if (ptr >= 0) {
        row_id = csc_row_idx[ptr];
    }
    // Process the specific clomun from matrix diagonal to top.
    // ptr indicates the index of csc_val corresponding to the column.
    while (ptr >= csc_col_ptr[col_id]) {
        // Get current NNZ's 2D location - row number.
        if (row_id == col_id) {
            // When thread handles the NNZ on the diagonal, it is responsible to 
            // update yi corresponding to current diagonal NNZ.
            __threadfence();
            // Only when the dependence has been decreased to 1 can the thread update yi.
            if (in_degree[col_id] == 1) {
                y[col_id] = (alpha * x[col_id] - tmp_sum[col_id]) / csc_val[ptr];
                ptr--;
                if (ptr >= 0) {
                    row_id = csc_row_idx[ptr];
                }
            }
        } else {
            // When thread handles the NNZ off the diagonal, it is responsible to update tmp_sum of its row.
            // Only when the diagonal NNZ has been processed by current thread, can the thread 
            // execute codes in this branch, and yi(y[col]) is also updated for the thread.
            atomicAdd(&tmp_sum[row_id], csc_val[ptr] * y[col_id]);
            __threadfence();
            // When temporary cumulative value has been written to global memory, 
            // the number of current row's dependence needs to be decreased.
            atomicSub(&in_degree[row_id], 1);
            ptr--;
            if (ptr >= 0) {
                row_id = csc_row_idx[ptr];
            }
        }
    }
    return;
}

// Similar to above kernel.
// This one is for conjuate matrix choice.
template<typename T, typename U>
__global__ static void
spsv_csc_n_up_conj_cw_kernel(
    const T* csc_col_ptr,
    const T* csc_row_idx,
    const U* csc_val,
    const T m,
    const U alpha,
    const U* x,
    U* y,
    T* id_extractor,
    U* tmp_sum,
    T* in_degree
) {
    return;
}

// hipFloatComplex Conjugate
template<typename T>
__global__ static void
spsv_csc_n_up_conj_cw_kernel(
    const T* csc_col_ptr,
    const T* csc_row_idx,
    const hipFloatComplex* csc_val,
    const T m,
    const hipFloatComplex alpha,
    const hipFloatComplex* x,
    hipFloatComplex* y,
    T* id_extractor,
    hipFloatComplex* tmp_sum,
    T* in_degree
) {
    T col_id = atomicAdd(id_extractor, 1);
    if (col_id >= m) {
        return;
    }
    col_id = m - 1 - col_id;
    T ptr = csc_col_ptr[col_id + 1] - 1;
    while (ptr > csc_col_ptr[col_id] && csc_row_idx[ptr] > col_id) {
        ptr--;
    }
    T row_id;
    if (ptr >= 0) {
        row_id = csc_row_idx[ptr];
    }
    while (ptr >= csc_col_ptr[col_id]) {
        if (row_id == col_id) {   // case 1: col_id == row_id
            __threadfence();
            if (in_degree[col_id] == 1) {
                y[col_id] = (alpha * x[col_id] - tmp_sum[col_id]) / csc_val[ptr];
                ptr--;
                if (ptr >= 0) {
                    row_id = csc_row_idx[ptr];
                }
            }
        } else {  // case 2: col_id != row_id
            atomicAdd(&tmp_sum[row_id], hipConjf(csc_val[ptr]) * y[col_id]);
            __threadfence();
            atomicSub(&in_degree[row_id], 1);
            ptr--;
            if (ptr >= 0) {
                row_id = csc_row_idx[ptr];
            }
        }
    }
    return;
}

// hipDoubleComplex Conjugate
template<typename T>
__global__ static void
spsv_csc_n_up_conj_cw_kernel(
    const T* csc_col_ptr,
    const T* csc_row_idx,
    const hipDoubleComplex* csc_val,
    const T m,
    const hipDoubleComplex alpha,
    const hipDoubleComplex* x,
    hipDoubleComplex* y,
    T* id_extractor,
    hipDoubleComplex* tmp_sum,
    T* in_degree
) {
    T col_id = atomicAdd(id_extractor, 1);
    if (col_id >= m) {
        return;
    }
    col_id = m - 1 - col_id;
    T ptr = csc_col_ptr[col_id + 1] - 1;
    while (ptr > csc_col_ptr[col_id] && csc_row_idx[ptr] > col_id) {
        ptr--;
    }
    T row_id;
    if (ptr >= 0) {
        row_id = csc_row_idx[ptr];
    }
    while (ptr >= csc_col_ptr[col_id]) {
        if (row_id == col_id) {  // case 1: col_id == row_id
            __threadfence();
            if (in_degree[col_id] == 1) { 
                y[col_id] = (alpha * x[col_id] - tmp_sum[col_id]) / csc_val[ptr];
                // y[col_id] = (alpha * x[col_id] - tmp_sum[col_id]) / hipConj(csc_val[ptr]);
                atomicSub(&in_degree[col_id], 1);
                ptr--;
                if (ptr >= 0) {
                    row_id = csc_row_idx[ptr];
                }
            }
        } else {  // case 2: col_id != row_id
            atomicAdd(&tmp_sum[row_id], hipConj(csc_val[ptr]) * y[col_id]);
            // atomicAdd(&tmp_sum[row_id], hipConj(csc_val[ptr]) * y[col_id]);
            __threadfence();
            atomicSub(&in_degree[row_id], 1);
            ptr--;
            if (ptr >= 0) {
                row_id = csc_row_idx[ptr];
            }
        }
    }
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csc_n_up_cw(
    alphasparseHandle_t handle,
    T m,
    T nnz,
    const U alpha,
    const U* csc_val,
    const T* csc_row_ind,
    const T* csc_col_ptr,
    const U* x,
    U* y,
    void *externalBuffer,
    bool is_conj
) {
    int threadPerBlock = 256;
    int blockPerGrid = (m - 1) / threadPerBlock + 1;

    U *tmp_sum = reinterpret_cast<U*>(externalBuffer);
    hipMemset(tmp_sum, {}, m * sizeof(U));

    T *in_degree = reinterpret_cast<T*>(reinterpret_cast<char*>(tmp_sum) + m * sizeof(U));
    hipMemset(in_degree, 0, m * sizeof(T));

    spsv_csc_n_up_cw_preprocess<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        csc_col_ptr,
        csc_row_ind,
        m,
        in_degree
    );

    hipDeviceSynchronize();

    T *id_extractor = reinterpret_cast<T*>(reinterpret_cast<char*>(in_degree) + m * sizeof(T));
    hipMemset(id_extractor, 0, sizeof(T));

    if (!is_conj) {
        // printf("is not conjudate\n");
        spsv_csc_n_up_cw_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
            csc_col_ptr,
            csc_row_ind,
            csc_val,
            m,
            alpha,
            x,
            y,
            id_extractor,
            tmp_sum,
            in_degree
        );
    } else {
        // printf("is conjugate\n");
        spsv_csc_n_up_conj_cw_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
            csc_col_ptr,
            csc_row_ind,
            csc_val,
            m,
            alpha,
            x,
            y,
            id_extractor,
            tmp_sum,
            in_degree
        );
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
