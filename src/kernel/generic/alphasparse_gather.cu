#include "alphasparse/handle.h"
#include <cuda_runtime.h>
#include "alphasparse/spapi.h"

#include "alphasparse.h"

#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/internal_check.h"

template <typename T, typename U>
alphasparseStatus_t gather_template(alphasparseHandle_t handle,
            const alphasparseDnVecDescr_t y,
            alphasparseSpVecDescr_t x
            )
{
const int threadPerBlock = 256;

    const T nnz          = x->nnz;
    const T blockPerGrid = min((T)2, ((T)threadPerBlock + nnz - (T)1) / threadPerBlock);
    const T *x_ind = (T *)x->idx_data;
    U *x_val = (U *)x->val_data;
    const U *y_val = (U *)y->values;
    gather_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(nnz, y_val, x_val, x_ind);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseGather(alphasparseHandle_t handle,
                       const alphasparseDnVecDescr_t y,
                       alphasparseSpVecDescr_t x)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check if descriptors are initialized
    if (x->init == false || y->init == false) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check for matching types while we do not support mixed precision computation
    if (x->data_type != y->data_type) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // single real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_32F) {
        return gather_template<int32_t, float>(handle, y, x);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_64F) {
        return gather_template<int32_t, double>(handle, y, x);
    }
    // // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_32F) {
        return gather_template<int32_t, cuFloatComplex>(handle, y, x);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_64F) {
        return gather_template<int32_t, cuDoubleComplex>(handle, y, x);
    }
    // half ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_16F) {
        return gather_template<int32_t, half>(handle, y, x);
    }
    // half complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_16F) {
        return gather_template<int32_t, half2>(handle, y, x);
    }
    #if (CUDA_ARCH >= 80)
    // bf16 ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_16BF) {
        return gather_template<int32_t, nv_bfloat16>(handle, y, x);
    }
    // bf16 complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_16BF) {
        return gather_template<int32_t, nv_bfloat162>(handle, y, x);
    }
    #endif
    // single real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_32F) {
        return gather_template<int64_t, float>(handle, y, x);
    }
    // double real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_64F) {
        return gather_template<int64_t, double>(handle, y, x);
    }
    // // single complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_32F) {
        return gather_template<int64_t, cuFloatComplex>(handle, y, x);
    }
    // double complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_64F) {
        return gather_template<int64_t, cuDoubleComplex>(handle, y, x);
    }
    // half ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_16F) {
        return gather_template<int64_t, half>(handle, y, x);
    }
    // half complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_16F) {
        return gather_template<int64_t, half2>(handle, y, x);
    }
    #if (CUDA_ARCH >= 80)
    // bf16 ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_16BF) {
        return gather_template<int64_t, nv_bfloat16>(handle, y, x);
    }
    // bf16 complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_16BF) {
        return gather_template<int64_t, nv_bfloat162>(handle, y, x);
    }
    #endif
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}


template <typename T, typename U>
__global__ static void
gather_device(
        const T nnz,
        const U *y,
        U *x_val,
        const T *x_ind
      )
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (T i = tid; i < nnz; i += stride) {
        x_val[i] = y[x_ind[i]];
    }
}
