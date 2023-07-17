#include "alphasparse.h"
#include <iostream>
#include "alphasparse_bsric02.h"

alphasparseStatus_t
alphasparseSbsric02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            float* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(float);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsric02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const float* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsric02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            double* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(double);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsric02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const double* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsric02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(float) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsric02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsric02_bufferSize(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = blockDim * mb * sizeof(double) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsric02_analysis(alphasparseHandle_t handle,
                        alphasparseDirection_t dirA,
                        int mb,
                        int nnzb,
                        const alphasparseMatDescr_t descrA,
                        const void* bsrValA,
                        const int* bsrRowPtrA,
                        const int* bsrColIndA,
                        int blockDim,
                        alpha_bsric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U>
alphasparseStatus_t
bsric02_template(alphasparseHandle_t handle,
                alphasparseDirection_t dirA,
                int mb,
                int nnzb,
                const alphasparseMatDescr_t descrA,
                void* bsrValA,
                const int* bsrRowPtrA,
                const int* bsrColIndA,
                int blockDim,
                alpha_bsric02Info_t info,
                alphasparseSolvePolicy_t policy,
                void* pBuffer)
{
    int * diag_pos = NULL;
    cudaMalloc(&diag_pos, sizeof(int) * (mb * blockDim));
    cudaMemset(diag_pos, 0, sizeof(int) * (mb * blockDim));
    int* d_done_array = NULL;
    cudaMalloc(&d_done_array, sizeof(int) * (mb * blockDim));
    // Initialize buffers
    cudaMemset(d_done_array, 0, sizeof(int) * (mb * blockDim));
    findDiag<<<2, 64, 0, handle->stream>>>(mb, bsrRowPtrA, bsrColIndA, diag_pos);
    bsric0_binsearch_kernel<32, 32, T><<<dim3(mb), dim3(32), 0, handle->stream>>>(mb, dirA, bsrRowPtrA, bsrColIndA, (T*)bsrValA, diag_pos, blockDim, d_done_array, ALPHA_SPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    float* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsric02_template<float, float>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    double* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsric02_template<double, double>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    void* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsric02_template<cuFloatComplex, float>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    void* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    bsric02_template<cuDoubleComplex, double>(handle, dirA, mb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseXbsric02_zeroPivot(alphasparseHandle_t handle,
                            alpha_bsric02Info_t info,
                            int* position)
{
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}