#include "alphasparse.h"
#include <iostream>
#include "alphasparse_gpsv.h"

alphasparseStatus_t
alphasparseSgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                                int algo,
                                                int m,
                                                const float* ds,
                                                const float* dl,
                                                const float* d,
                                                const float* du,
                                                const float* dw,
                                                const float* x,
                                                int batchCount,
                                                size_t* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = sizeof(float) * ((m * batchCount - 1) / 256 + 1) * 256 * 3;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                                int algo,
                                                int m,
                                                const double* ds,
                                                const double* dl,
                                                const double* d,
                                                const double* du,
                                                const double* dw,
                                                const double* x,
                                                int batchCount,
                                                size_t* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = sizeof(double) * ((m * batchCount - 1) / 256 + 1) * 256 * 3;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                                int algo,
                                                int m,
                                                const void* ds,
                                                const void* dl,
                                                const void* d,
                                                const void* du,
                                                const void* dw,
                                                const void* x,
                                                int batchCount,
                                                size_t* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = 2 * sizeof(float) * ((m * batchCount - 1) / 256 + 1) * 256 * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                                int algo,
                                                int m,
                                                const void* ds,
                                                const void* dl,
                                                const void* d,
                                                const void* du,
                                                const void* dw,
                                                const void* x,
                                                int batchCount,
                                                size_t* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = 2 * sizeof(double) * ((m * batchCount - 1) / 256 + 1) * 256 * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
gpsvInterleavedBatch_template(alphasparseHandle_t handle,
                            int algo,
                            int m,
                            void* ds,
                            void* dl,
                            void* d,
                            void* du,
                            void* dw,
                            void* x,
                            int batchCount,
                            void* pBuffer)
{
    if(m < 3) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    char* ptr = reinterpret_cast<char*>(pBuffer);
    int   batchStride = batchCount;
    if(std::is_same<T, float>() || std::is_same<T, double>())
    {
        T* dt1 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batchCount - 1) / 256 + 1) * 256;

        T* dt2 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batchCount - 1) / 256 + 1) * 256;

        T* B = reinterpret_cast<T*>(ptr);

        // Initialize buffers with zero
        cudaMemsetAsync(dt1, 0, sizeof(T) * m * batchCount, handle->stream);
        cudaMemsetAsync(dt2, 0, sizeof(T) * m * batchCount, handle->stream);

    #define GPSV_DIM 256
        dim3 gpsv_blocks((batchCount - 1) / GPSV_DIM + 1);
        dim3 gpsv_threads(GPSV_DIM);

        // Copy strided B into buffer
        gpsv_strided_gather<GPSV_DIM><<<
                           gpsv_blocks,
                           gpsv_threads,
                           0,
                           handle->stream>>>(
                           m,
                           batchCount,
                           batchStride,
                           (T*)x,
                           B);

        // Launch kernel
        gpsv_interleaved_batch_householder_qr_kernel<GPSV_DIM><<<
                           gpsv_blocks,
                           gpsv_threads,
                           0,
                           handle->stream>>>(
                           m,
                           batchCount,
                           batchStride,
                           (T*)ds,
                           (T*)dl,
                           (T*)d,
                           (T*)du,
                           (T*)dw,
                           (T*)x,
                           dt1,
                           dt2,
                           B);
    #undef GPSV_DIM
    }
    else
    {
        T* r3 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batchCount - 1) / 256 + 1) * 256;
        T* r4 = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((m * batchCount - 1) / 256 + 1) * 256;

        cudaMemsetAsync(
            r3, 0, sizeof(T) * ((m * batchCount - 1) / 256 + 1) * 256, handle->stream);
        cudaMemsetAsync(
            r4, 0, sizeof(T) * ((m * batchCount - 1) / 256 + 1) * 256, handle->stream);

       gpsv_interleaved_batch_givens_qr_kernel<128, T><<<
                           dim3(((batchCount - 1) / 128 + 1), 1, 1),
                           dim3(128, 1, 1),
                           0,
                           handle->stream>>>(
                           m,
                           batchCount,
                           batchStride,
                           (T*)ds,
                           (T*)dl,
                           (T*)d,
                           (T*)du,
                           (T*)dw,
                           r3,
                           r4,
                           (T*)x);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSgpsvInterleavedBatch(alphasparseHandle_t handle,
                                int algo,
                                int m,
                                float* ds,
                                float* dl,
                                float* d,
                                float* du,
                                float* dw,
                                float* x,
                                int batchCount,
                                void* pBuffer)
{
    return gpsvInterleavedBatch_template<float>(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

alphasparseStatus_t
alphasparseDgpsvInterleavedBatch(alphasparseHandle_t handle,
                                int algo,
                                int m,
                                double* ds,
                                double* dl,
                                double* d,
                                double* du,
                                double* dw,
                                double* x,
                                int batchCount,
                                void* pBuffer)
{
    return gpsvInterleavedBatch_template<double>(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

alphasparseStatus_t
alphasparseCgpsvInterleavedBatch(alphasparseHandle_t handle,
                                int algo,
                                int m,
                                void* ds,
                                void* dl,
                                void* d,
                                void* du,
                                void* dw,
                                void* x,
                                int batchCount,
                                void* pBuffer)
{
    return gpsvInterleavedBatch_template<cuFloatComplex>(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

alphasparseStatus_t
alphasparseZgpsvInterleavedBatch(alphasparseHandle_t handle,
                                int algo,
                                int m,
                                void* ds,
                                void* dl,
                                void* d,
                                void* du,
                                void* dw,
                                void* x,
                                int batchCount,
                                void* pBuffer)
{
    return gpsvInterleavedBatch_template<cuDoubleComplex>(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}