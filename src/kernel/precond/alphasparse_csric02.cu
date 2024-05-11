#include "alphasparse.h"
#include <iostream>
#include "alphasparse_csric02.h"

alphasparseStatus_t
alphasparseScsric02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            float* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(float);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsric02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const float* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsric02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            double* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(double);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsric02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const double* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsric02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            void* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(float) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsric02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const void* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsric02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            void* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csric02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(double) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsric02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const void* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csric02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
csric02_template(alphasparseHandle_t handle,
                int m,
                int nnz,
                const alphasparseMatDescr_t descrA,
                void* csrValA_valM,
                const int* csrRowPtrA,
                const int* csrColIndA,
                alpha_csric02Info_t info,
                alphasparseSolvePolicy_t policy,
                void* pBuffer)
{
    int * diag_pos = NULL;
    cudaMalloc(&diag_pos, sizeof(int) * (m+1));
    cudaMemset(diag_pos, 0, sizeof(int) * (m+1));
    int* d_done_array = NULL;
    cudaMalloc(&d_done_array, sizeof(int) * (m+1));
    // Initialize buffers
    cudaMemset(d_done_array, 0, sizeof(int) * m);
    findDiag<<<2, 64, 0, handle->stream>>>(m, csrRowPtrA, csrColIndA, diag_pos);
#define CSRIC0_DIM 256
    dim3 csric0_blocks((m * 32 - 1) / CSRIC0_DIM + 1);
    dim3 csric0_threads(CSRIC0_DIM);
    csric0_binsearch_kernel<CSRIC0_DIM, 32, T><<<csric0_blocks, csric0_threads, 0, handle->stream>>>(m, csrRowPtrA, csrColIndA, (T*)csrValA_valM, diag_pos, d_done_array, ALPHA_SPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
#undef CSRIC0_DIM
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    float* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csric02_template<float>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    double* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csric02_template<double>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    void* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csric02_template<cuFloatComplex>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    void* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csric02_template<cuDoubleComplex>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}