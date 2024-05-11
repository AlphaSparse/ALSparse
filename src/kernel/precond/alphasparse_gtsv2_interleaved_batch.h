/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "alphasparse.h"
#include "gtsv_interleaved_batch_device.h"

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             cudaGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
             __LINE__,                                                         \
             cusparseGetErrorString(status),                                   \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

template<typename T>
alphasparseStatus_t
gtsv_interleaved_batch_buffer_size_template(alphasparseHandle_t handle,
                                            int alg,
                                            int m,
                                            const T* dl,
                                            const T* d,
                                            const T* du,
                                            const T* x,
                                            int batch_count,
                                            int batch_stride,
                                            size_t* buffer_size)
{
  // Check for valid handle and matrix descriptor
  if (handle == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
  }

  // Check sizes
  if (m <= 1 || batch_count < 0 || batch_stride < batch_count) {
    return ALPHA_SPARSE_STATUS_INVALID_SIZE;
  }

  // Check for valid buffer_size pointer
  if (buffer_size == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  // Quick return if possible
  if (batch_count == 0) {
    *buffer_size = 0;
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }

  // Check pointer arguments
  if (dl == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (d == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (du == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (x == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  switch (alg) {
    case 0: {
      *buffer_size = 0;
      *buffer_size +=
        ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dc1
      *buffer_size +=
        ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dx1
      break;
    }
    case 1: {
      *buffer_size = 0;
      *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // u2
      *buffer_size +=
        ((sizeof(int) * m * batch_count - 1) / 256 + 1) * 256; // p
      break;
    }
    case 2: {
      *buffer_size = 0;
      *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // r2
      break;
    }
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_interleaved_batch_thomas_template(alphasparseHandle_t handle,
                                       int m,
                                       T* dl,
                                       T* d,
                                       T* du,
                                       T* x,
                                       int batch_count,
                                       int batch_stride,
                                       void* temp_buffer)
{
  char* ptr = reinterpret_cast<char*>(temp_buffer);
  T* dc1 = reinterpret_cast<T*>(temp_buffer);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* dx1 = reinterpret_cast<T*>(reinterpret_cast<void*>(ptr));
  //  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  gtsv_interleaved_batch_thomas_kernel<256>
    <<<dim3(((batch_count - 1) / 256 + 1), 1, 1),
       dim3(256, 1, 1),
       0,
       handle->stream>>>(m, batch_count, batch_stride, dl, d, du, dc1, dx1, x);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_interleaved_batch_lu_template(alphasparseHandle_t handle,
                                   int m,
                                   T* dl,
                                   T* d,
                                   T* du,
                                   T* x,
                                   int batch_count,
                                   int batch_stride,
                                   void* temp_buffer)
{
  char* ptr = reinterpret_cast<char*>(temp_buffer);
  T* u2 = reinterpret_cast<T*>(temp_buffer);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  int* p = reinterpret_cast<int*>(reinterpret_cast<void*>(ptr));
  // ptr += ((sizeof(int) * m * batch_count - 1) / 256 + 1) * 256;

  CHECK_CUDA(
    cudaMemsetAsync(u2,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));

  gtsv_interleaved_batch_lu_kernel<128>
    <<<dim3(((batch_count - 1) / 128 + 1), 1, 1),
       dim3(128, 1, 1),
       0,
       handle->stream>>>(m, batch_count, batch_stride, dl, d, du, u2, p, x);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_interleaved_batch_qr_template(alphasparseHandle_t handle,
                                   int m,
                                   T* dl,
                                   T* d,
                                   T* du,
                                   T* x,
                                   int batch_count,
                                   int batch_stride,
                                   void* temp_buffer)
{
  char* ptr = reinterpret_cast<char*>(temp_buffer);
  T* r2 = reinterpret_cast<T*>(ptr);
  //   ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

  CHECK_CUDA(
    cudaMemsetAsync(r2,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));

  gtsv_interleaved_batch_qr_kernel<128>
    <<<dim3(((batch_count - 1) / 128 + 1), 1, 1),
       dim3(128, 1, 1),
       0,
       handle->stream>>>(m, batch_count, batch_stride, dl, d, du, r2, x);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_interleaved_batch_template(alphasparseHandle_t handle,
                                int alg,
                                int m,
                                T* dl,
                                T* d,
                                T* du,
                                T* x,
                                int batch_count,
                                int batch_stride,
                                void* temp_buffer)
{
  // Check for valid handle and matrix descriptor
  if (handle == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
  }

  // Check sizes
  if (m <= 1 || batch_count < 0 || batch_stride < batch_count) {
    return ALPHA_SPARSE_STATUS_INVALID_SIZE;
  }

  // Quick return if possible
  if (batch_count == 0) {
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }

  // Check pointer arguments
  if (dl == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (d == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (du == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (x == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (temp_buffer == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  switch (alg) {
    case 0:
      return gtsv_interleaved_batch_thomas_template(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
    case 1:
      return gtsv_interleaved_batch_lu_template(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
    case 2:
      return gtsv_interleaved_batch_qr_template(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

// /*
//  * ===========================================================================
//  *    C wrapper
//  * ===========================================================================
//  */
// #define C_IMPL(NAME, TYPE) \
//     extern "C" alphasparseStatus_t NAME(alphasparseHandle_t handle,        \
//                                      int alg, \
//                                      int                  m,             \
//                                      const TYPE*                    dl, \
//                                      const TYPE*                    d, \
//                                      const TYPE*                    du, \
//                                      const TYPE*                    x, \
//                                      int                  batch_count,   \
//                                      int                  batch_stride,  \
//                                      size_t* buffer_size)   \
//     try \
//     { \
//         return gtsv_interleaved_batch_buffer_size_template( \
//             handle, alg, m, dl, d, du, x, batch_count, batch_stride,
//             buffer_size); \
//     } \
//     catch(...) \
//     { \
//         return exception_to_alphasparseStatus_t(); \
//     }

// C_IMPL(sgtsv_interleaved_batch_buffer_size, float);
// C_IMPL(dgtsv_interleaved_batch_buffer_size, double);
// C_IMPL(cgtsv_interleaved_batch_buffer_size,
// float_complex);
// C_IMPL(zgtsv_interleaved_batch_buffer_size,
// double_complex);

// #undef C_IMPL

// #define C_IMPL(NAME, TYPE)                                                         \
//     extern "C" alphasparseStatus_t NAME(alphasparseHandle_t               handle,        \
//                                      int alg,           \
//                                      int                  m,             \
//                                      TYPE*                          dl,            \
//                                      TYPE*                          d,             \
//                                      TYPE*                          du,            \
//                                      TYPE*                          x,             \
//                                      int                  batch_count,   \
//                                      int                  batch_stride,  \
//                                      void*                          temp_buffer)   \
//     try                                                                            \
//     {                                                                              \
//         return gtsv_interleaved_batch_template(                          \
//             handle, alg, m, dl, d, du, x, batch_count, batch_stride, temp_buffer); \
//     }                                                                              \
//     catch(...)                                                                     \
//     {                                                                              \
//         return exception_to_alphasparseStatus_t();                                    \
//     }

// C_IMPL(sgtsv_interleaved_batch, float);
// C_IMPL(dgtsv_interleaved_batch, double);
// C_IMPL(cgtsv_interleaved_batch, float_complex);
// C_IMPL(zgtsv_interleaved_batch, double_complex);

// #undef C_IMPL
