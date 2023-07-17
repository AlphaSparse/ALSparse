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
#include "gtsv2_device.h"

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
gtsv_buffer_size_template(alphasparseHandle_t handle,
                          int m,
                          int n,
                          const T* dl,
                          const T* d,
                          const T* du,
                          const T* B,
                          int ldb,
                          size_t* buffer_size)
{
  // Check for valid handle and matrix descriptor
  if (handle == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
  }

  // Check sizes
  if (m <= 1 || n < 0 || ldb < std::max(1, m)) {
    return ALPHA_SPARSE_STATUS_INVALID_SIZE;
  }

  // Check for valid buffer_size pointer
  if (buffer_size == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  // Quick return if possible
  if (n == 0) {
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
  } else if (B == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  constexpr unsigned int BLOCKSIZE = 256;

  int block_dim = 2;
  int m_pad = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
  int gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
  while (gridsize > 512) {
    block_dim *= 2;
    m_pad = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
    gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
  }

  // round up to next power of 2
  gridsize = fnp2(gridsize);

  *buffer_size = 0;

  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // dl_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // d_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // du_pad
  *buffer_size += ((sizeof(T) * m_pad * n - 1) / 256 + 1) * 256; // rhs_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // w_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // v_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // w2_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // v2_pad
  *buffer_size += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;     // mt_pad

  *buffer_size +=
    ((sizeof(T) * 2 * gridsize * n - 1) / 256 + 1) * 256; // rhs_scratch
  *buffer_size += ((sizeof(T) * 2 * gridsize - 1) / 256 + 1) * 256; // w_scratch
  *buffer_size += ((sizeof(T) * 2 * gridsize - 1) / 256 + 1) * 256; // v_scratch

  *buffer_size += ((sizeof(int) * m_pad - 1) / 256 + 1) * 256; // pivot_pad

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<unsigned int BLOCKSIZE, unsigned int BLOCKDIM, typename T>
alphasparseStatus_t
gtsv_spike_solver_template(alphasparseHandle_t handle,
                           int m,
                           int n,
                           int m_pad,
                           int gridsize,
                           const T* dl,
                           const T* d,
                           const T* du,
                           T* B,
                           int ldb,
                           void* temp_buffer)
{
  char* ptr = reinterpret_cast<char*>(temp_buffer);
  T* dl_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* d_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* du_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* rhs_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad * n - 1) / 256 + 1) * 256;
  T* w_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* v_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* w2_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* v2_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;
  T* mt_pad = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * m_pad - 1) / 256 + 1) * 256;

  T* rhs_scratch = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * 2 * gridsize * n - 1) / 256 + 1) * 256;
  T* w_scratch = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * 2 * gridsize - 1) / 256 + 1) * 256;
  T* v_scratch = reinterpret_cast<T*>(ptr);
  ptr += ((sizeof(T) * 2 * gridsize - 1) / 256 + 1) * 256;

  int* pivot_pad = reinterpret_cast<int*>(ptr);
  //    ptr += ((sizeof(int) * m_pad - 1) / 256 + 1) * 256;
  gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3((m_pad - 1) / BLOCKSIZE + 1), dim3(BLOCKSIZE), 0, handle->stream>>>(
      m, m_pad, m_pad, dl, dl_pad, T{});
  gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3((m_pad - 1) / BLOCKSIZE + 1), dim3(BLOCKSIZE), 0, handle->stream>>>(
      m, m_pad, m_pad, d, d_pad, make_value<T>(1.f));
  gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3((m_pad - 1) / BLOCKSIZE + 1), dim3(BLOCKSIZE), 0, handle->stream>>>(
      m, m_pad, m_pad, du, du_pad, T{});
  gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3((m_pad - 1) / BLOCKSIZE + 1, n),
       dim3(BLOCKSIZE),
       0,
       handle->stream>>>(m, m_pad, ldb, B, rhs_pad, T{});

  CHECK_CUDA(cudaMemsetAsync(w_pad, 0, m_pad * sizeof(T), handle->stream));
  CHECK_CUDA(cudaMemsetAsync(v_pad, 0, m_pad * sizeof(T), handle->stream));
  gtsv_LBM_wv_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3(gridsize), dim3(BLOCKSIZE), 0, handle->stream>>>(
      m_pad, n, ldb, dl_pad, d_pad, du_pad, w_pad, v_pad, mt_pad, pivot_pad);

  if (n % 8 == 0) {
    gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 8>
      <<<dim3(gridsize, n / 8), dim3(BLOCKSIZE), 0, handle->stream>>>(
        m_pad, n, ldb, dl_pad, d_pad, du_pad, rhs_pad, mt_pad, pivot_pad);
  } else if (n % 4 == 0) {
    gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 4>
      <<<dim3(gridsize, n / 4), dim3(BLOCKSIZE), 0, handle->stream>>>(
        m_pad, n, ldb, dl_pad, d_pad, du_pad, rhs_pad, mt_pad, pivot_pad);
  } else if (n % 2 == 0) {
    gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 2>
      <<<dim3(gridsize, n / 2), dim3(BLOCKSIZE), 0, handle->stream>>>(
        m_pad, n, ldb, dl_pad, d_pad, du_pad, rhs_pad, mt_pad, pivot_pad);
  } else {
    gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 1>
      <<<dim3(gridsize, n), dim3(BLOCKSIZE), 0, handle->stream>>>(
        m_pad, n, ldb, dl_pad, d_pad, du_pad, rhs_pad, mt_pad, pivot_pad);
  }

  CHECK_CUDA(cudaMemcpyAsync(w2_pad,
                             w_pad,
                             m_pad * sizeof(T),
                             cudaMemcpyDeviceToDevice,
                             handle->stream));
  CHECK_CUDA(cudaMemcpyAsync(v2_pad,
                             v_pad,
                             m_pad * sizeof(T),
                             cudaMemcpyDeviceToDevice,
                             handle->stream));

  gtsv_spike_block_level_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3(gridsize, n), dim3(BLOCKSIZE), 0, handle->stream>>>(m_pad,
                                                                n,
                                                                ldb,
                                                                rhs_pad,
                                                                w_pad,
                                                                v_pad,
                                                                w2_pad,
                                                                v2_pad,
                                                                rhs_scratch,
                                                                w_scratch,
                                                                v_scratch);

  // gridsize is always a power of 2
  if (gridsize == 2) {
    gtsv_solve_spike_grid_level_kernel<2>
      <<<dim3(1, n), dim3(2), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 4) {
    gtsv_solve_spike_grid_level_kernel<4>
      <<<dim3(1, n), dim3(4), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 8) {
    gtsv_solve_spike_grid_level_kernel<8>
      <<<dim3(1, n), dim3(8), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 16) {
    gtsv_solve_spike_grid_level_kernel<16>
      <<<dim3(1, n), dim3(16), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 32) {
    gtsv_solve_spike_grid_level_kernel<32>
      <<<dim3(1, n), dim3(32), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 64) {
    gtsv_solve_spike_grid_level_kernel<64>
      <<<dim3(1, n), dim3(64), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 128) {
    gtsv_solve_spike_grid_level_kernel<128>
      <<<dim3(1, n), dim3(128), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 256) {
    gtsv_solve_spike_grid_level_kernel<256>
      <<<dim3(1, n), dim3(256), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  } else if (gridsize == 512) {
    gtsv_solve_spike_grid_level_kernel<512>
      <<<dim3(1, n), dim3(512), 0, handle->stream>>>(
        m_pad, n, ldb, rhs_scratch, w_scratch, v_scratch);
  }

  gtsv_solve_spike_propagate_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3(gridsize, n), dim3(BLOCKSIZE), 0, handle->stream>>>(
      m_pad, n, ldb, rhs_pad, w2_pad, v2_pad, rhs_scratch);
  gtsv_spike_backward_substitution_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3(gridsize, n), dim3(BLOCKSIZE), 0, handle->stream>>>(
      m_pad, n, ldb, rhs_pad, w2_pad, v2_pad);
  gtsv_transpose_back_array_kernel<BLOCKSIZE, BLOCKDIM>
    <<<dim3((m_pad - 1) / BLOCKSIZE + 1, n),
       dim3(BLOCKSIZE),
       0,
       handle->stream>>>(m, m_pad, ldb, rhs_pad, B);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gtsv_template(alphasparseHandle_t handle,
              int m,
              int n,
              const T* dl,
              const T* d,
              const T* du,
              T* B,
              int ldb,
              void* temp_buffer)
{
  // Check for valid handle and matrix descriptor
  if (handle == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
  }

  // Check sizes
  if (m <= 1 || n < 0 || ldb < std::max(1, m)) {
    return ALPHA_SPARSE_STATUS_INVALID_SIZE;
  }

  // Quick return if possible
  if (n == 0) {
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }

  // Check pointer arguments
  if (dl == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (d == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (du == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (B == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  } else if (temp_buffer == nullptr) {
    return ALPHA_SPARSE_STATUS_INVALID_POINTER;
  }

  constexpr unsigned int BLOCKSIZE = 256;

  int block_dim = 2;
  int m_pad = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
  int gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
  while (gridsize > 512) {
    block_dim *= 2;
    m_pad = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
    gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
  }

  // round up to next power of 2
  gridsize = fnp2(gridsize);

  if (block_dim == 2) {
    return gtsv_spike_solver_template<BLOCKSIZE, 2>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 4) {
    return gtsv_spike_solver_template<BLOCKSIZE, 4>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 8) {
    return gtsv_spike_solver_template<BLOCKSIZE, 8>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 16) {
    return gtsv_spike_solver_template<BLOCKSIZE, 16>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 32) {
    return gtsv_spike_solver_template<BLOCKSIZE, 32>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 64) {
    return gtsv_spike_solver_template<BLOCKSIZE, 64>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 128) {
    return gtsv_spike_solver_template<BLOCKSIZE, 128>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else if (block_dim == 256) {
    return gtsv_spike_solver_template<BLOCKSIZE, 256>(
      handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
  } else {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
}

// /*
//  * ===========================================================================
//  *    C wrapper
//  * ===========================================================================
//  */
// #define C_IMPL(NAME, TYPE) \
//     extern "C" alphasparseStatus_t NAME(alphasparseHandle_t handle, \
//                                      int    m, \
//                                      int    n, \
//                                      const TYPE*      dl, \
//                                      const TYPE*      d, \
//                                      const TYPE*      du, \
//                                      const TYPE*      B, \
//                                      int    ldb, \
//                                      size_t*          buffer_size) \
//     try \
//     { \
//         return gtsv_buffer_size_template(handle, m, n, dl, d, du,
//         B, ldb, buffer_size); \
//     } \
//     catch(...) \
//     { \
//         return exception_to_alphasparseStatus_t(); \
//     }

// C_IMPL(sgtsv_buffer_size, float);
// C_IMPL(dgtsv_buffer_size, double);
// C_IMPL(cgtsv_buffer_size, float_complex);
// C_IMPL(zgtsv_buffer_size, double_complex);

// #undef C_IMPL

// #define C_IMPL(NAME, TYPE)                                                            \
//     extern "C" alphasparseStatus_t NAME(alphasparseHandle_t handle,                         \
//                                      int    m,                              \
//                                      int    n,                              \
//                                      const TYPE*      dl,                             \
//                                      const TYPE*      d,                              \
//                                      const TYPE*      du,                             \
//                                      TYPE*            B,                              \
//                                      int    ldb,                            \
//                                      void*            temp_buffer)                    \
//     try                                                                               \
//     {                                                                                 \
//         return gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer); \
//     }                                                                                 \
//     catch(...)                                                                        \
//     {                                                                                 \
//         return exception_to_alphasparseStatus_t();                                       \
//     }

// C_IMPL(sgtsv, float);
// C_IMPL(dgtsv, double);
// C_IMPL(cgtsv, float_complex);
// C_IMPL(zgtsv, double_complex);

// #undef C_IMPL
