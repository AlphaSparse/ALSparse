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
#include "alphasparse/types.h" 
#include "gtsv2_nopivot_strided_batch_device.h"

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

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_STAGE1(                     \
  T, block_size, stride, iter)                                                 \
  gtsv_nopivot_strided_batch_pcr_pow2_stage1_kernel<block_size>                \
    <<<dim3(((m - 1) / block_size + 1), batch_count, 1),                       \
       dim3(block_size, 1, 1),                                                 \
       0,                                                                      \
       handle->stream>>>(                                                      \
      stride,                                                                  \
      m,                                                                       \
      batch_count,                                                             \
      ((iter == 0) ? batch_stride : m),                                        \
      ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),                    \
      ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),                     \
      ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),                    \
      ((iter == 0) ? x : (((iter & 1) == 0) ? drhs0 : drhs1)),                 \
      (((iter & 1) == 0) ? da1 : da0),                                         \
      (((iter & 1) == 0) ? db1 : db0),                                         \
      (((iter & 1) == 0) ? dc1 : dc0),                                         \
      (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_STAGE2(T, block_size, iter)  \
  gtsv_nopivot_strided_batch_cr_pow2_stage2_kernel<block_size>                 \
    <<<dim3(subsystem_count, batch_count, 1),                                  \
       dim3(block_size),                                                       \
       0,                                                                      \
       handle->stream>>>(m,                                                    \
                         batch_count,                                          \
                         batch_stride,                                         \
                         (((iter & 1) != 0) ? da1 : da0),                      \
                         (((iter & 1) != 0) ? db1 : db0),                      \
                         (((iter & 1) != 0) ? dc1 : dc0),                      \
                         (((iter & 1) != 0) ? drhs1 : drhs0),                  \
                         x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_SHARED(T, block_size)        \
  gtsv_nopivot_strided_batch_cr_pow2_shared_kernel<block_size>                 \
    <<<dim3(batch_count), dim3(block_size), 0, handle->stream>>>(              \
      m, batch_count, batch_stride, dl, d, du, x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, block_size)       \
  gtsv_nopivot_strided_batch_pcr_pow2_shared_kernel<block_size>                \
    <<<dim3(batch_count), dim3(block_size), 0, handle->stream>>>(              \
      m, batch_count, batch_stride, dl, d, du, x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(                   \
  T, block_size, pcr_size)                                                     \
  gtsv_nopivot_strided_batch_crpcr_pow2_shared_kernel<block_size, pcr_size>    \
    <<<dim3(batch_count), dim3(block_size), 0, handle->stream>>>(              \
      m, batch_count, batch_stride, dl, d, du, x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, block_size)            \
  gtsv_nopivot_strided_batch_pcr_shared_kernel<block_size>                     \
    <<<dim3(batch_count), dim3(block_size), 0, handle->stream>>>(              \
      m, batch_count, batch_stride, dl, d, du, x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE1(                          \
  T, block_size, stride, iter)                                                 \
  gtsv_nopivot_strided_batch_pcr_stage1_kernel<block_size>                     \
    <<<dim3(((m - 1) / block_size + 1), batch_count, 1),                       \
       dim3(block_size),                                                       \
       0,                                                                      \
       handle->stream>>>(                                                      \
      stride,                                                                  \
      m,                                                                       \
      batch_count,                                                             \
      ((iter == 0) ? batch_stride : m),                                        \
      ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),                    \
      ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),                     \
      ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),                    \
      ((iter == 0) ? x : (((iter & 1) == 0) ? drhs0 : drhs1)),                 \
      (((iter & 1) == 0) ? da1 : da0),                                         \
      (((iter & 1) == 0) ? db1 : db0),                                         \
      (((iter & 1) == 0) ? dc1 : dc0),                                         \
      (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE2(T, block_size, iter)      \
  gtsv_nopivot_strided_batch_pcr_stage2_kernel<block_size>                     \
    <<<dim3(subsystem_count, batch_count, 1),                                  \
       dim3(block_size),                                                       \
       0,                                                                      \
       handle->stream>>>(m,                                                    \
                         batch_count,                                          \
                         batch_stride,                                         \
                         (((iter & 1) != 0) ? da1 : da0),                      \
                         (((iter & 1) != 0) ? db1 : db0),                      \
                         (((iter & 1) != 0) ? dc1 : dc0),                      \
                         (((iter & 1) != 0) ? drhs1 : drhs0),                  \
                         x);

template<typename T>
alphasparseStatus_t
gtsv_no_pivot_strided_batch_buffer_size_template(
  alphasparseHandle_t handle,
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
  if (m <= 1 || batch_count < 0 || batch_stride < m) {
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

  if (m <= 512) {
    *buffer_size = 0;
  } else {
    *buffer_size = 0;

    *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // da0
    *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // da1
    *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // db0
    *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // db1
    *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dc0
    *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dc1
    *buffer_size +=
      ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // drhs0
    *buffer_size +=
      ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // drhs1
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_no_pivot_strided_batch_small_template(alphasparseHandle_t handle,
                                                     int m,
                                                     const T* dl,
                                                     const T* d,
                                                     const T* du,
                                                     T* x,
                                                     int batch_count,
                                                     int batch_stride,
                                                     void* temp_buffer)
{
  assert(m <= 512);

  // Run special algorithm if m is power of 2
  if ((m & (m - 1)) == 0) {
    if (m == 2) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 2);
    } else if (m == 4) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 4);
    } else if (m == 8) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 8);
    } else if (m == 16) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 16);
    } else if (m == 32) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 32);
    // } else if (m == 64) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 64);
    // } else if (m == 128) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, 64, 64);
    // } else if (m == 256) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, 128, 64);
    // } else if (m == 512) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, 256, 64);
    }
  } else {
    if (m <= 4) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 4);
    } else if (m <= 8) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 8);
    } else if (m <= 16) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 16);
    } else if (m <= 32) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 32);
    // } else if (m <= 64) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 64);
    // } else if (m <= 128) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 128);
    // } else if (m <= 256) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 256);
    // } else if (m <= 512) {
    //   LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 512);
    }
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_no_pivot_strided_batch_large_template(alphasparseHandle_t handle,
                                                     int m,
                                                     const T* dl,
                                                     const T* d,
                                                     const T* du,
                                                     T* x,
                                                     int batch_count,
                                                     int batch_stride,
                                                     void* temp_buffer)
{
  assert(m > 512);

  char* ptr = reinterpret_cast<char*>(temp_buffer);
  T* da0 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* da1 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* db0 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* db1 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* dc0 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* dc1 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* drhs0 = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
  T* drhs1 = reinterpret_cast<T*>(ptr);
  // ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

  CHECK_CUDA(
    cudaMemsetAsync(da0,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(da1,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(db0,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(db1,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(dc0,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(dc1,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(drhs0,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));
  CHECK_CUDA(
    cudaMemsetAsync(drhs1,
                    0,
                    ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256,
                    handle->stream));

  // Run special algorithm if m is power of 2
  if ((m & (m - 1)) == 0) {
    // Stage1: Break large tridiagonal system into multiple smaller systems
    // using parallel cyclic reduction so that each sub system is of size 512.
    int iter =
      static_cast<int>(log2((float)m)) - static_cast<int>(log2((float)512));

    int stride = 1;
    for (int i = 0; i < iter; i++) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_STAGE1(T, 256, stride, i);

      stride *= 2;
    }

    // Stage2: Solve the many systems from stage1 in parallel using cyclic
    // reduction.
    int subsystem_count = 1 << iter;

    LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_STAGE2(T, 256, iter);
  } else {
    // Stage1: Break large tridiagonal system into multiple smaller systems
    // using parallel cyclic reduction so that each sub system is of size 512 or
    // less.
    int iter =
      static_cast<int>(log2((float)m)) - static_cast<int>(log2((float)512)) + 1;

    int stride = 1;
    for (int i = 0; i < iter; i++) {
      LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE1(T, 256, stride, i);

      stride *= 2;
    }

    // Stage2: Solve the many systems from stage1 in parallel using cyclic
    // reduction.
    int subsystem_count = 1 << iter;

    LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE2(T, 512, iter);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_no_pivot_strided_batch_template(alphasparseHandle_t handle,
                                               int m,
                                               const T* dl,
                                               const T* d,
                                               const T* du,
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
  if (m <= 1 || batch_count < 0 || batch_stride < m) {
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
  } else if (m > 512 && temp_buffer == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  // If m is small we can solve the systems entirely in shared memory
  if (m <= 512) {
    return gtsv_no_pivot_strided_batch_small_template(
      handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
  }

  return gtsv_no_pivot_strided_batch_large_template(
    handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

// /*
//  * ===========================================================================
//  *    C wrapper
//  * ===========================================================================
//  */
// #define C_IMPL(NAME, TYPE) \
//     extern "C" alphasparseStatus_t NAME(alphasparseHandle_t handle, \
//                                      int    m, \
//                                      const TYPE*      dl, \
//                                      const TYPE*      d, \
//                                      const TYPE*      du, \
//                                      const TYPE*      x, \
//                                      int    batch_count, \
//                                      int    batch_stride, \
//                                      size_t*          buffer_size) \
//     try \
//     { \
//         return gtsv_no_pivot_strided_batch_buffer_size_template( \
//             handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
//             \
//     } \
//     catch(...) \
//     { \
//         return exception_to_alphasparseStatus_t(); \
//     }

// C_IMPL(sgtsv_no_pivot_strided_batch_buffer_size, float);
// C_IMPL(dgtsv_no_pivot_strided_batch_buffer_size, double);
// C_IMPL(cgtsv_no_pivot_strided_batch_buffer_size,
// float_complex);
// C_IMPL(zgtsv_no_pivot_strided_batch_buffer_size,
// double_complex);

// #undef C_IMPL

// #define C_IMPL(NAME, TYPE)                                                    \
//     extern "C" alphasparseStatus_t NAME(alphasparseHandle_t handle,                 \
//                                      int    m,                      \
//                                      const TYPE*      dl,                     \
//                                      const TYPE*      d,                      \
//                                      const TYPE*      du,                     \
//                                      TYPE*            x,                      \
//                                      int    batch_count,            \
//                                      int    batch_stride,           \
//                                      void*            temp_buffer)            \
//     try                                                                       \
//     {                                                                         \
//         return gtsv_no_pivot_strided_batch_template(                \
//             handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer); \
//     }                                                                         \
//     catch(...)                                                                \
//     {                                                                         \
//         return exception_to_alphasparseStatus_t();                               \
//     }

// C_IMPL(sgtsv_no_pivot_strided_batch, float);
// C_IMPL(dgtsv_no_pivot_strided_batch, double);
// C_IMPL(cgtsv_no_pivot_strided_batch, float_complex);
// C_IMPL(zgtsv_no_pivot_strided_batch, double_complex);

// #undef C_IMPL
