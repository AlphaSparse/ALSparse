#include "alphasparse.h"
#include <iostream>
#include "alphasparse_bsrilu02.h"

alphasparseStatus_t
alphasparseSbsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_bsrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            float* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_bsrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            double* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_bsrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            void* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_bsrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            void* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrilu02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            float* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(float);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrilu02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const float* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrilu02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            double* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(double);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrilu02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const double* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrilu02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(float) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrilu02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrilu02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(double) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrilu02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U>
alphasparseStatus_t
bsrilu02_template(alphasparseHandle_t handle,
                alphasparseDirection_t dirA,
                int mb,
                int nnzb,
                const alphasparseMatDescr_t descrA,
                void* bsrValA,
                const int* bsrRowPtrA,
                const int* bsrColIndA,
                int blockDim,
                alpha_bsrilu02Info_t info,
                alphasparseSolvePolicy_t policy,
                void* pBuffer)
{
    int * diag_pos = NULL;
    U boost_tol = {};
    T boost_val = {};
    int enable_boost = 0;
    cudaMalloc(&diag_pos, sizeof(int) * (mb * blockDim));
    cudaMemset(diag_pos, 0, sizeof(int) * (mb * blockDim));
    int* d_done_array = NULL;
    cudaMalloc(&d_done_array, sizeof(int) * (mb * blockDim));
    // Initialize buffers
    cudaMemset(d_done_array, 0, sizeof(int) * (mb * blockDim));
    findDiag<<<2, 64, 0, handle->stream>>>(mb, bsrRowPtrA, bsrColIndA, diag_pos);
    bsrilu0_general_kernel<128, 32, T, U><<<dim3((32 * mb - 1) / 128 + 1), dim3(128), 0, handle->stream>>>(mb, dirA, bsrRowPtrA, bsrColIndA, (T*)bsrValA, diag_pos, blockDim, d_done_array, ALPHA_SPARSE_INDEX_BASE_ZERO, enable_boost, boost_tol, boost_val);
    cudaDeviceSynchronize();
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsrilu02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    float* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsrilu02_template<float, float>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsrilu02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    double* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsrilu02_template<double, double>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsrilu02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    void* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsrilu02_template<cuFloatComplex, float>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsrilu02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    void* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsrilu02_template<cuDoubleComplex, double>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseXbsrilu02_zeroPivot(alphasparseHandle_t handle,
                            alpha_bsrilu02Info_t info,
                            int* position)
{
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}