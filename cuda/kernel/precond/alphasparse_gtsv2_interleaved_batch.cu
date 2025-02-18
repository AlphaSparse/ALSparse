#include "alphasparse_gtsv2_interleaved_batch.h"
#include <iostream>

alphasparseStatus_t
alphasparseSgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const float* dl,
                                               const float* d,
                                               const float* du,
                                               const float* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes)
{
  gtsv_interleaved_batch_buffer_size_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const double* dl,
                                               const double* d,
                                               const double* du,
                                               const double* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes)
{
  gtsv_interleaved_batch_buffer_size_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const cuFloatComplex* dl,
                                               const cuFloatComplex* d,
                                               const cuFloatComplex* du,
                                               const cuFloatComplex* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes)
{
  gtsv_interleaved_batch_buffer_size_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const cuDoubleComplex* dl,
                                               const cuDoubleComplex* d,
                                               const cuDoubleComplex* du,
                                               const cuDoubleComplex* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes)
{
  gtsv_interleaved_batch_buffer_size_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 float* dl,
                                 float* d,
                                 float* du,
                                 float* x,
                                 int batchCount,
                                 void* pBuffer)
{
  gtsv_interleaved_batch_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 double* dl,
                                 double* d,
                                 double* du,
                                 double* x,
                                 int batchCount,
                                 void* pBuffer)
{
  gtsv_interleaved_batch_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 cuFloatComplex* dl,
                                 cuFloatComplex* d,
                                 cuFloatComplex* du,
                                 cuFloatComplex* x,
                                 int batchCount,
                                 void* pBuffer)
{
  gtsv_interleaved_batch_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 cuDoubleComplex* dl,
                                 cuDoubleComplex* d,
                                 cuDoubleComplex* du,
                                 cuDoubleComplex* x,
                                 int batchCount,
                                 void* pBuffer)
{
  gtsv_interleaved_batch_template(
    handle, algo, m, dl, d, du, x, batchCount, batchCount, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}