#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <iostream>
#include "alphasparse_csrilu02.h"

alphasparseStatus_t
alphasparseScsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_csrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            float* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_csrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            double* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_csrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            void* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsrilu02_numericBoost(alphasparseHandle_t handle,
                            alpha_csrilu02Info_t info,
                            int enable_boost,
                            double* tol,
                            void* boost_val)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsrilu02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            float* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(float);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsrilu02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const float* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsrilu02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            double* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(double);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsrilu02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const double* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsrilu02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            void* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(float) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsrilu02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const void* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsrilu02_bufferSize(alphasparseHandle_t handle,
                            int m,
                            int nnz,
                            const alphasparseMatDescr_t descrA,
                            void* csrValA,
                            const int* csrRowPtrA,
                            const int* csrColIndA,
                            alpha_csrilu02Info_t info,
                            int* pBufferSizeInBytes)
{
    * pBufferSizeInBytes = m * sizeof(double) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsrilu02_analysis(alphasparseHandle_t handle,
                        int m,
                        int nnz,
                        const alphasparseMatDescr_t descrA,
                        const void* csrValA,
                        const int* csrRowPtrA,
                        const int* csrColIndA,
                        alpha_csrilu02Info_t info,
                        alphasparseSolvePolicy_t policy,
                        void* pBuffer)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U>
alphasparseStatus_t
csrilu02_template(alphasparseHandle_t handle,
                int m,
                int nnz,
                const alphasparseMatDescr_t descrA,
                void* csrValA_valM,
                const int* csrRowPtrA,
                const int* csrColIndA,
                alpha_csrilu02Info_t info,
                alphasparseSolvePolicy_t policy,
                void* pBuffer)
{
    int * diag_pos = NULL;
    U boost_tol = {};
    T boost_val = {};
    int enable_boost = 0;
    cudaMalloc(&diag_pos, sizeof(int) * (m+1));
    cudaMemset(diag_pos, 0, sizeof(int) * (m+1));
    int* d_done_array = NULL;
    cudaMalloc(&d_done_array, sizeof(int) * (m+1));
    // Initialize buffers
    cudaMemset(d_done_array, 0, sizeof(int) * m);
    findDiag<<<2, 64, 0, handle->stream>>>(m, csrRowPtrA, csrColIndA, diag_pos);
#define csrilu0_DIM 256
    dim3 csrilu0_blocks((m * 32 - 1) / csrilu0_DIM + 1);
    dim3 csrilu0_threads(csrilu0_DIM);
    csrilu0_binsearch_kernel<csrilu0_DIM, 32, T, U><<<csrilu0_blocks, csrilu0_threads, 0, handle->stream>>>(m, csrRowPtrA, csrColIndA, (T*)csrValA_valM, diag_pos, d_done_array, ALPHA_SPARSE_INDEX_BASE_ZERO, enable_boost, boost_tol, boost_val);
    cudaDeviceSynchronize();
#undef csrilu0_DIM
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsrilu02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    float* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csrilu02_template<float, float>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsrilu02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    double* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csrilu02_template<double, double>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsrilu02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    void* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csrilu02_template<cuFloatComplex, float>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsrilu02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    void* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csrilu02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    csrilu02_template<cuDoubleComplex, double>(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}