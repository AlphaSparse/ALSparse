#include "alphasparse_gtsv2_strided_batch.h"
#include <iostream>

alphasparseStatus_t
alphasparseSgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const float* dl,
                                            const float* d,
                                            const float* du,
                                            const float* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes)
{
  gtsv_no_pivot_strided_batch_buffer_size_template<float>(
    handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const double* dl,
                                            const double* d,
                                            const double* du,
                                            const double* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes)
{
  gtsv_no_pivot_strided_batch_buffer_size_template<double>(
    handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const cuFloatComplex* dl,
                                            const cuFloatComplex* d,
                                            const cuFloatComplex* du,
                                            const cuFloatComplex* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes)
{
  gtsv_no_pivot_strided_batch_buffer_size_template<cuFloatComplex>(
    handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes)
{
  gtsv_no_pivot_strided_batch_buffer_size_template<cuDoubleComplex>(
    handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const float* dl,
                              const float* d,
                              const float* du,
                              float* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer)
{
  gtsv_no_pivot_strided_batch_template<float>(
    handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const double* dl,
                              const double* d,
                              const double* du,
                              double* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer)
{
  gtsv_no_pivot_strided_batch_template<double>(
    handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const cuFloatComplex* dl,
                              const cuFloatComplex* d,
                              const cuFloatComplex* du,
                              cuFloatComplex* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer)
{
  gtsv_no_pivot_strided_batch_template<cuFloatComplex>(
    handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const cuDoubleComplex* dl,
                              const cuDoubleComplex* d,
                              const cuDoubleComplex* du,
                              cuDoubleComplex* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer)
{
  gtsv_no_pivot_strided_batch_template<cuDoubleComplex>(
    handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
