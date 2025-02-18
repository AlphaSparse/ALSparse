#include "alphasparse/handle.h"
#include "alphasparse/spapi.h"
#include <cuda_runtime.h>

#include "alphasparse.h"
#include "alphasparse/types.h" 

#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/internal_check.h"

template <typename T, typename U>
alphasparseStatus_t scatter_template(alphasparseHandle_t handle,
            const alphasparseSpVecDescr_t x,
            alphasparseDnVecDescr_t y)
{
    const int threadPerBlock = 256;
    const T nnz          = x->nnz;
    const T blockPerGrid = min((T)2, ((T)threadPerBlock + nnz - (T)1) / threadPerBlock);
    const T *x_ind = (T *)x->idx_data;
    const U *x_val = (U *)x->val_data;
    U *y_val = (U *)y->values;
    scatter_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(nnz, x_val, x_ind, y_val);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScatter(alphasparseHandle_t handle,
                        const alphasparseSpVecDescr_t x,
                        alphasparseDnVecDescr_t y)
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
        return scatter_template<int32_t, float>(handle, x, y);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_64F) {
        return scatter_template<int32_t, double>(handle, x, y);
    }
    // // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_32F) {
        return scatter_template<int32_t, cuFloatComplex>(handle, x, y);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_64F) {
        return scatter_template<int32_t, cuDoubleComplex>(handle, x, y);
    }
    // half ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_16F) {
        return scatter_template<int32_t, half>(handle, x, y);
    }
    // half complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_16F) {
        return scatter_template<int32_t, half2>(handle, x, y);
    }
    #if (CUDA_ARCH >= 80)
    // bf16 ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_16BF) {
        return scatter_template<int32_t, nv_bfloat16>(handle, x, y);
    }
    // bf16 complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_16BF) {
        return scatter_template<int32_t, nv_bfloat162>(handle, x, y);
    }
    #endif
    // single real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_32F) {
        return scatter_template<int64_t, float>(handle, x, y);
    }
    // double real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_64F) {
        return scatter_template<int64_t, double>(handle, x, y);
    }
    // // single complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_32F) {
        return scatter_template<int64_t, cuFloatComplex>(handle, x, y);
    }
    // double complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_64F) {
        return scatter_template<int64_t, cuDoubleComplex>(handle, x, y);
    }
    // half ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_16F) {
        return scatter_template<int64_t, half>(handle, x, y);
    }
    // half complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_16F) {
        return scatter_template<int64_t, half2>(handle, x, y);
    }
    #if (CUDA_ARCH >= 80)
    // bf16 ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_16BF) {
        return scatter_template<int64_t, nv_bfloat16>(handle, x, y);
    }
    // bf16 complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_16BF) {
        return scatter_template<int64_t, nv_bfloat162>(handle, x, y);
    }
    #endif
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

template <typename T, typename U>
__global__ static void
scatter_device(
        const T nnz,
      const U *x_val,
      const T *x_ind,
        U *y)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (T i = tid; i < nnz; i += stride) {
        y[x_ind[i]] = x_val[i];
    }
}

#define INSTANTIATE(T, U)                                                      \
  template __global__ static void scatter_device<T, U>(                           \
    const T nnz, const U *x_val, const T *x_ind, U *y)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, half);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, half2);
INSTANTIATE(int32_t, cuFloatComplex);
INSTANTIATE(int32_t, cuDoubleComplex);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, half);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, half2);
INSTANTIATE(int64_t, cuFloatComplex);
INSTANTIATE(int64_t, cuDoubleComplex);
