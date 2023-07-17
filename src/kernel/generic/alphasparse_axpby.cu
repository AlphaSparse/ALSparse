#include "alphasparse/handle.h"
#include <cuda_runtime.h>
#include <iostream>
#include "alphasparse/util/auxiliary.h"

#include "alphasparse.h"

#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/internal_check.h"

const int threadPerBlock = 256;

template <typename T, typename U, typename V>
alphasparseStatus_t axpby_template(alphasparseHandle_t handle,
            const void *alpha,
            const alphasparseSpVecDescr_t x,
            const void *beta,
            alphasparseDnVecDescr_t y)
{
    const T size          = x->size;
    const T nnz          = x->nnz;
    const T blockPerGrid = min((T)2, ((T)threadPerBlock + nnz - (T)1) / threadPerBlock);
    const T *x_ind = (T *)x->idx_data;
    const U *x_val = (U *)x->val_data;
    U *y_val = (U *)y->values;
    const V alpha_val = *(V*)alpha;
    const V beta_val = *(V*)beta;
    axpby_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(size, nnz, alpha_val, x_val, beta_val, x_ind, y_val);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseAxpby(alphasparseHandle_t handle,
                      const void *alpha,
                      const alphasparseSpVecDescr_t x,
                      const void *beta,
                      alphasparseDnVecDescr_t y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }
    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr) {
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
        return axpby_template<int32_t, float, float>(handle, alpha, x, beta, y);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_64F) {
        return axpby_template<int32_t, double, double>(handle, alpha, x, beta, y);
    }
    // // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_32F) {
        return axpby_template<int32_t, cuFloatComplex, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_64F) {
        return axpby_template<int32_t, cuDoubleComplex, cuDoubleComplex>(handle, alpha, x, beta, y);
    }
    // half ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_16F) {
        return axpby_template<int32_t, half, float>(handle, alpha, x, beta, y);
    }
    // half complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_16F) {
        return axpby_template<int32_t, half2, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    #if (CUDA_ARCH >= 80)
    // bf16 ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_16BF) {
        return axpby_template<int32_t, nv_bfloat16, float>(handle, alpha, x, beta, y);
    }
    // bf16 complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_16BF) {
        return axpby_template<int32_t, nv_bfloat162, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    #endif
    // single real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_32F) {
        return axpby_template<int64_t, float, float>(handle, alpha, x, beta, y);
    }
    // double real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_64F) {
        return axpby_template<int64_t, double, double>(handle, alpha, x, beta, y);
    }
    // // single complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_32F) {
        return axpby_template<int64_t, cuFloatComplex, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    // double complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_64F) {
        return axpby_template<int64_t, cuDoubleComplex, cuDoubleComplex>(handle, alpha, x, beta, y);
    }
    // half ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_16F) {
        return axpby_template<int64_t, half, float>(handle, alpha, x, beta, y);
    }
    // half complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_16F) {
        return axpby_template<int64_t, half2, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    #if (CUDA_ARCH >= 80)
    // bf16 ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_16BF) {
        return axpby_template<int64_t, nv_bfloat16, float>(handle, alpha, x, beta, y);
    }
    // bf16 complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_16BF) {
        return axpby_template<int64_t, nv_bfloat162, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    #endif
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

template <typename T, typename U, typename V>
__global__ static void
axpby_device(const T size,
        const T nnz,
      const V alpha,
      const U *x_val,
      const V beta,
      const T *x_ind,
      U *y)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (T i = tid; i < size; i += stride) {
        y[i] = beta * y[i];
    }
    for (T i = tid; i < nnz; i += stride) {
        y[x_ind[i]] = alpha * x_val[i] + y[x_ind[i]];
    }
}
