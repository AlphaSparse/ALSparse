#include "alphasparse.h"
#include "alphasparse_gtsv2_nopivot.h"
#include <iostream>

alphasparseStatus_t
alphasparseSgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const float* dl,
                                const float* d,
                                const float* du,
                                float* B,
                                int ldb,
                                size_t* bufferSizeInBytes)
{
  gtsv2_no_pivot_buffer_size_template<float>(
    handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const double* dl,
                                const double* d,
                                const double* du,
                                double* B,
                                int ldb,
                                size_t* bufferSizeInBytes)
{
  gtsv2_no_pivot_buffer_size_template<double>(
    handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const cuFloatComplex* dl,
                                const cuFloatComplex* d,
                                const cuFloatComplex* du,
                                cuFloatComplex* B,
                                int ldb,
                                size_t* bufferSizeInBytes)
{
  gtsv2_no_pivot_buffer_size_template<cuFloatComplex>(
    handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const cuDoubleComplex* dl,
                                const cuDoubleComplex* d,
                                const cuDoubleComplex* du,
                                cuDoubleComplex* B,
                                int ldb,
                                size_t* bufferSizeInBytes)
{
  gtsv2_no_pivot_buffer_size_template<cuDoubleComplex>(
    handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSgtsv2_nopivot(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const float* dl,
                  const float* d,
                  const float* du,
                  float* B,
                  int ldb,
                  void* pBuffer)
{
  gtsv2_no_pivot_template<float>(handle, m, n, dl, d, du, B, ldb, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgtsv2_nopivot(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const double* dl,
                  const double* d,
                  const double* du,
                  double* B,
                  int ldb,
                  void* pBuffer)
{
  gtsv2_no_pivot_template<double>(handle, m, n, dl, d, du, B, ldb, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgtsv2_nopivot(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const cuFloatComplex* dl,
                  const cuFloatComplex* d,
                  const cuFloatComplex* du,
                  cuFloatComplex* B,
                  int ldb,
                  void* pBuffer)
{
  gtsv2_no_pivot_template<cuFloatComplex>(handle, m, n, dl, d, du, B, ldb, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgtsv2_nopivot(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const cuDoubleComplex* dl,
                  const cuDoubleComplex* d,
                  const cuDoubleComplex* du,
                  cuDoubleComplex* B,
                  int ldb,
                  void* pBuffer)
{
  gtsv2_no_pivot_template<cuDoubleComplex>(handle, m, n, dl, d, du, B, ldb, pBuffer);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}