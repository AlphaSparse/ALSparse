#include "alphasparse/handle.h"
#include "alphasparse/spapi.h"
#include <cuda_runtime.h>

#include "alphasparse.h"
#include "alphasparse/util/internal_check.h"

#include <iostream>

template<typename T, typename U>
alphasparseStatus_t
spvv_template(alphasparseHandle_t handle,
              const alphasparseSpVecDescr_t x,
              const alphasparseDnVecDescr_t y,
              U* result,
              alphasparseDataType datatype)
{
  const int threadPerBlock = 256;
  const T nnz = x->nnz;
  const int blockPerGrid =
    min((T)32, (threadPerBlock + nnz - (T)1) / threadPerBlock);

  U *dev_part_c, *part_c, *dev_result;
  part_c = (U*)malloc(sizeof(U) * blockPerGrid);
  cudaMalloc((void**)&dev_part_c, sizeof(U) * blockPerGrid);
  cudaMalloc((void**)&dev_result, sizeof(U));

  const T* x_ind = (T*)x->idx_data;
  U* x_val = (U*)x->val_data;
  U* y_val = (U*)y->values;

  spvv_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(
    nnz, x_val, x_ind, y_val, dev_part_c);
  cudaMemcpy(
    part_c, dev_part_c, sizeof(U) * blockPerGrid, cudaMemcpyDeviceToHost);
  reduce_result<<<dim3(1), dim3(1)>>>(blockPerGrid, dev_part_c, dev_result);
  cudaMemcpy(result, dev_result, sizeof(U), cudaMemcpyDeviceToHost);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpvv(alphasparseHandle_t handle,
                 alphasparseOperation_t trans,
                 const alphasparseSpVecDescr_t x,
                 const alphasparseDnVecDescr_t y,
                 void* result,
                 alphasparseDataType compute_type,
                 void* temp_buffer)
{
  // Check for valid handle and matrix descriptor
  if (handle == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
  }

  //
  // Check the rest of pointer arguments
  //
  if (x == nullptr || y == nullptr || result == nullptr ||
      temp_buffer == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  if (x->init == false || y->init == false)
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;

  // int8 int32 real ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_R_8I &&
      compute_type == ALPHA_R_32I) {
    int8_t* x_val = (int8_t*)malloc(sizeof(int8_t) * x->nnz);
    int8_t* y_val = (int8_t*)malloc(sizeof(int8_t) * x->size);
    cudaMemcpy(
      x_val, x->val_data, sizeof(int8_t) * x->nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      y_val, y->values, sizeof(int8_t) * x->size, cudaMemcpyDeviceToHost);
    int32_t* x_val2 = (int32_t*)malloc(sizeof(int32_t) * x->nnz);
    int32_t* y_val2 = (int32_t*)malloc(sizeof(int32_t) * x->size);
    for (int i = 0; i < x->nnz; i++) {
      x_val2[i] = x_val[i];
    }
    for (int i = 0; i < x->size; i++) {
      y_val2[i] = y_val[i];
    }
    int32_t* dx_val = NULL;
    int32_t* dy = NULL;
    cudaMalloc((void**)&dx_val, sizeof(int32_t) * x->nnz);
    cudaMalloc((void**)&dy, sizeof(int32_t) * x->size);
    cudaMemcpy(
      dx_val, x_val2, sizeof(int32_t) * x->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y_val2, sizeof(int32_t) * x->size, cudaMemcpyHostToDevice);
    alphasparseSpVecDescr_t x2{};
    alphasparseCreateSpVec(&x2,
                                   x->size,
                                   x->nnz,
                                   x->idx_data,
                                   (void*)dx_val,
                                   ALPHA_SPARSE_INDEXTYPE_I32,
                                   ALPHA_SPARSE_INDEX_BASE_ZERO,
                                   ALPHA_R_8I);
    alphasparseDnVecDescr_t y2{};
    alphasparseCreateDnVec(
      &y2, y->size, (void*)dy, ALPHA_R_8I);
    alphasparseStatus_t status = spvv_template<int32_t, int32_t>(
      handle, x2, y2, (int32_t*)result, ALPHA_R_32I);
    return status;
  }
  // int8 float real ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_R_8I &&
      compute_type == ALPHA_R_32F) {
    int8_t* x_val = (int8_t*)malloc(sizeof(int8_t) * x->nnz);
    int8_t* y_val = (int8_t*)malloc(sizeof(int8_t) * x->size);
    cudaMemcpy(
      x_val, x->val_data, sizeof(int8_t) * x->nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      y_val, y->values, sizeof(int8_t) * x->size, cudaMemcpyDeviceToHost);
    float* x_val2 = (float*)malloc(sizeof(float) * x->nnz);
    float* y_val2 = (float*)malloc(sizeof(float) * x->size);
    for (int i = 0; i < x->nnz; i++) {
      x_val2[i] = x_val[i];
    }
    for (int i = 0; i < x->size; i++) {
      y_val2[i] = y_val[i];
    }
    float* dx_val = NULL;
    float* dy = NULL;
    cudaMalloc((void**)&dx_val, sizeof(float) * x->nnz);
    cudaMalloc((void**)&dy, sizeof(float) * x->size);
    cudaMemcpy(
      dx_val, x_val2, sizeof(float) * x->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y_val2, sizeof(float) * x->size, cudaMemcpyHostToDevice);
    alphasparseSpVecDescr_t x2{};
    alphasparseCreateSpVec(&x2,
                                   x->size,
                                   x->nnz,
                                   x->idx_data,
                                   (void*)dx_val,
                                   ALPHA_SPARSE_INDEXTYPE_I32,
                                   ALPHA_SPARSE_INDEX_BASE_ZERO,
                                   ALPHA_R_8I);
    alphasparseDnVecDescr_t y2{};
    alphasparseCreateDnVec(
      &y2, y->size, (void*)dy, ALPHA_R_8I);
    alphasparseStatus_t status = spvv_template<int32_t, float>(
      handle, x2, y2, (float*)result, ALPHA_R_32I);
    return status;
  }
  // single real ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_R_8I) {
    return spvv_template<int32_t, float>(
      handle, x, y, (float*)result, ALPHA_R_8I);
  }
  // double real ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_R_64F) {
    return spvv_template<int32_t, double>(
      handle, x, y, (double*)result, ALPHA_R_64F);
  }
  // // single complex ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_C_32F) {
    return spvv_template<int32_t, cuFloatComplex>(
      handle,
      x,
      y,
      (cuFloatComplex*)result,
      ALPHA_C_32F);
  }
  // double complex ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_C_64F) {
    return spvv_template<int32_t, cuDoubleComplex>(
      handle,
      x,
      y,
      (cuDoubleComplex*)result,
      ALPHA_C_64F);
  }
  // half ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_R_16F) {
    half* result_converted = (half*)malloc(sizeof(half));
    alphasparseStatus_t status = spvv_template<int32_t, half>(
      handle, x, y, result_converted, ALPHA_R_16F);
    *(float*)result = __half2float(*result_converted);
    return status;
  }
  // half complex ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_C_16F) {
    half2* result_converted = (half2*)malloc(sizeof(half2));
    alphasparseStatus_t status = spvv_template<int32_t, half2>(
      handle, x, y, result_converted, ALPHA_C_16F);
    *(cuFloatComplex*)result = __half22complex(*result_converted);
    return status;
  }
  #if (CUDA_ARCH >= 80)
  // bf16 ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_R_16BF) {
    nv_bfloat16* result_converted = (nv_bfloat16*)malloc(sizeof(nv_bfloat16));
    alphasparseStatus_t status = spvv_template<int32_t, nv_bfloat16>(
      handle, x, y, result_converted, ALPHA_R_16BF);
    *(float*)result = __bfloat162float(*result_converted);
    return status;
  }
  // bf16 complex ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      x->data_type == ALPHA_C_16BF) {
    nv_bfloat162* result_converted = (nv_bfloat162*)malloc(sizeof(nv_bfloat162));
    alphasparseStatus_t status = spvv_template<int32_t, nv_bfloat162>(
      handle, x, y, result_converted, ALPHA_C_16BF);
    *(cuFloatComplex*)result = __bfloat1622complex(*result_converted);
    return status;
  }
  #endif
  // int8 real ; i32
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_R_8I &&
      compute_type == ALPHA_R_32I) {
    int8_t* x_val = (int8_t*)malloc(sizeof(int8_t) * x->nnz);
    int8_t* y_val = (int8_t*)malloc(sizeof(int8_t) * x->size);
    cudaMemcpy(
      x_val, x->val_data, sizeof(int8_t) * x->nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      y_val, y->values, sizeof(int8_t) * x->size, cudaMemcpyDeviceToHost);
    int32_t* x_val2 = (int32_t*)malloc(sizeof(int32_t) * x->nnz);
    int32_t* y_val2 = (int32_t*)malloc(sizeof(int32_t) * x->size);
    for (int i = 0; i < x->nnz; i++) {
      x_val2[i] = x_val[i];
    }
    for (int i = 0; i < x->size; i++) {
      y_val2[i] = y_val[i];
    }
    int32_t* dx_val = NULL;
    int32_t* dy = NULL;
    cudaMalloc((void**)&dx_val, sizeof(int32_t) * x->nnz);
    cudaMalloc((void**)&dy, sizeof(int32_t) * x->size);
    cudaMemcpy(
      dx_val, x_val2, sizeof(int32_t) * x->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y_val2, sizeof(int32_t) * x->size, cudaMemcpyHostToDevice);
    alphasparseSpVecDescr_t x2{};
    alphasparseCreateSpVec(&x2,
                                   x->size,
                                   x->nnz,
                                   x->idx_data,
                                   (void*)dx_val,
                                   ALPHA_SPARSE_INDEXTYPE_I32,
                                   ALPHA_SPARSE_INDEX_BASE_ZERO,
                                   ALPHA_R_8I);
    alphasparseDnVecDescr_t y2{};
    alphasparseCreateDnVec(
      &y2, y->size, (void*)dy, ALPHA_R_8I);
    alphasparseStatus_t status = spvv_template<int64_t, int32_t>(
      handle, x2, y2, (int32_t*)result, ALPHA_R_32I);
    return status;
  }
  // int8 float real ; i64
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_R_8I &&
      compute_type == ALPHA_R_32F) {
    int8_t* x_val = (int8_t*)malloc(sizeof(int8_t) * x->nnz);
    int8_t* y_val = (int8_t*)malloc(sizeof(int8_t) * x->size);
    cudaMemcpy(
      x_val, x->val_data, sizeof(int8_t) * x->nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      y_val, y->values, sizeof(int8_t) * x->size, cudaMemcpyDeviceToHost);
    float* x_val2 = (float*)malloc(sizeof(float) * x->nnz);
    float* y_val2 = (float*)malloc(sizeof(float) * x->size);
    for (int i = 0; i < x->nnz; i++) {
      x_val2[i] = x_val[i];
    }
    for (int i = 0; i < x->size; i++) {
      y_val2[i] = y_val[i];
    }
    float* dx_val = NULL;
    float* dy = NULL;
    cudaMalloc((void**)&dx_val, sizeof(float) * x->nnz);
    cudaMalloc((void**)&dy, sizeof(float) * x->size);
    cudaMemcpy(
      dx_val, x_val2, sizeof(float) * x->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y_val2, sizeof(float) * x->size, cudaMemcpyHostToDevice);
    alphasparseSpVecDescr_t x2{};
    alphasparseCreateSpVec(&x2,
                                   x->size,
                                   x->nnz,
                                   x->idx_data,
                                   (void*)dx_val,
                                   ALPHA_SPARSE_INDEXTYPE_I32,
                                   ALPHA_SPARSE_INDEX_BASE_ZERO,
                                   ALPHA_R_8I);
    alphasparseDnVecDescr_t y2{};
    alphasparseCreateDnVec(
      &y2, y->size, (void*)dy, ALPHA_R_8I);
    alphasparseStatus_t status = spvv_template<int64_t, float>(
      handle, x2, y2, (float*)result, ALPHA_R_32I);
    return status;
  }
  // single real ; i64  
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_R_8I) {
    return spvv_template<int64_t, float>(
      handle, x, y, (float*)result, ALPHA_R_8I);
  }
  // double real ; i64
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_R_64F) {
    return spvv_template<int64_t, double>(
      handle, x, y, (double*)result, ALPHA_R_64F);
  }
  // // single complex ; i64
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_C_32F) {
    return spvv_template<int64_t, cuFloatComplex>(
      handle,
      x,
      y,
      (cuFloatComplex*)result,
      ALPHA_C_32F);
  }
  // double complex ; i64
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_C_64F) {
    return spvv_template<int64_t, cuDoubleComplex>(
      handle,
      x,
      y,
      (cuDoubleComplex*)result,
      ALPHA_C_64F);
  }
  // half ; i64
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 &&
      x->data_type == ALPHA_R_16F) {
    half* result_converted = (half*)malloc(sizeof(half));
    alphasparseStatus_t status = spvv_template<int64_t, half>(
      handle, x, y, result_converted, ALPHA_R_16F);
    *(float*)result = __half2float(*result_converted);
    return status;
  }
  // half complex ; i64
  if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type ==
  ALPHA_C_16F) {
    half2* result_converted = (half2*)malloc(sizeof(half2));
    alphasparseStatus_t status = spvv_template<int64_t, half2>(
      handle, x, y, result_converted, ALPHA_C_16F);
    *(cuFloatComplex*)result = __half22complex(*result_converted);
    return status;
  }

  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

alphasparseStatus_t
alphasparseSpvv_buffersize(alphasparseHandle_t handle,
                            alphasparseOperation_t trans,
                            const alphasparseSpVecDescr_t x,
                            const alphasparseDnVecDescr_t y,
                            void* result,
                            alphasparseDataType compute_type,
                            size_t* buffer_size)
{
  // We do not need a buffer
  *buffer_size = 4;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U>
__global__ static void
spvv_device(T nnz, const U* x_val, const T* x_ind, const U* y, U* result)
{
  const int threadPerBlock = 256;
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // *
  int stride = gridDim.x * blockDim.x;
  int cacheidx = threadIdx.x;

  __shared__ U
    cache[threadPerBlock]; // block内线程共享__shared__，注意这里没有初始化

  U tmp = {};

  for (int i = idx; i < nnz; i += stride) // 每个线程先求和自己可以接触到的数
  {
    tmp += x_val[i] * y[x_ind[i]];
  }
  cache[cacheidx] = tmp;
  __syncthreads();

  // 规约
  T i = threadPerBlock / 2;
  while (i != 0) {
    if (cacheidx < i) // 只需要线程号小于i的线程参与计算
    {
      cache[cacheidx] += cache[cacheidx + i]; // 两两求和
    }
    i /= 2; // 循环变量
    __syncthreads();
  }
  if (cacheidx == 0) // 块内0号线程提交块内规约结果
    result[blockIdx.x] = cache[0];
}

template<typename U>
__global__ static void
reduce_result(int blockPerGrid, U* dev_part_c, U* dev_result)
{
  *dev_result = U{};
  for (int i = 0; i < blockPerGrid; i++) {
    *dev_result += dev_part_c[i];
  }
}

#define INSTANTIATE(T, U)                                                      \
  template __global__ static void spvv_device<T, U>(                           \
    T nnz, const U* x_val, const T* x_ind, const U* y, U* result)

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

#undef INSTANTIATE
