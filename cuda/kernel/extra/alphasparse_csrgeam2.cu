#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse_csrgeam2.h"
#include <iostream>

#define WFSIZE      64
#define CSRGEAM_DIM 256
#define BLOCKSIZE   CSRGEAM_DIM

alphasparseStatus_t
alphasparseScsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                    int m,
                                    int n,
                                    const float* alpha,
                                    const alphasparseMatDescr_t descrA,
                                    int nnzA,
                                    const float* csrSortedValA,
                                    const int* csrSortedRowPtrA,
                                    const int* csrSortedColIndA,
                                    const float* beta,
                                    const alphasparseMatDescr_t descrB,
                                    int nnzB,
                                    const float* csrSortedValB,
                                    const int* csrSortedRowPtrB,
                                    const int* csrSortedColIndB,
                                    const alphasparseMatDescr_t descrC,
                                    const float* csrSortedValC,
                                    const int* csrSortedRowPtrC,
                                    const int* csrSortedColIndC,
                                    size_t* pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4 * sizeof(float);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                    int m,
                                    int n,
                                    const double* alpha,
                                    const alphasparseMatDescr_t descrA,
                                    int nnzA,
                                    const double* csrSortedValA,
                                    const int* csrSortedRowPtrA,
                                    const int* csrSortedColIndA,
                                    const double* beta,
                                    const alphasparseMatDescr_t descrB,
                                    int nnzB,
                                    const double* csrSortedValB,
                                    const int* csrSortedRowPtrB,
                                    const int* csrSortedColIndB,
                                    const alphasparseMatDescr_t descrC,
                                    const double* csrSortedValC,
                                    const int* csrSortedRowPtrC,
                                    const int* csrSortedColIndC,
                                    size_t* pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4 * sizeof(double);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                    int m,
                                    int n,
                                    const void* alpha,
                                    const alphasparseMatDescr_t descrA,
                                    int nnzA,
                                    const void* csrSortedValA,
                                    const int* csrSortedRowPtrA,
                                    const int* csrSortedColIndA,
                                    const void* beta,
                                    const alphasparseMatDescr_t descrB,
                                    int nnzB,
                                    const void* csrSortedValB,
                                    const int* csrSortedRowPtrB,
                                    const int* csrSortedColIndB,
                                    const alphasparseMatDescr_t descrC,
                                    const void* csrSortedValC,
                                    const int* csrSortedRowPtrC,
                                    const int* csrSortedColIndC,
                                    size_t* pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4 * sizeof(float) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                    int m,
                                    int n,
                                    const void* alpha,
                                    const alphasparseMatDescr_t descrA,
                                    int nnzA,
                                    const void* csrSortedValA,
                                    const int* csrSortedRowPtrA,
                                    const int* csrSortedColIndA,
                                    const void* beta,
                                    const alphasparseMatDescr_t descrB,
                                    int nnzB,
                                    const void* csrSortedValB,
                                    const int* csrSortedRowPtrB,
                                    const int* csrSortedColIndB,
                                    const alphasparseMatDescr_t descrC,
                                    const void* csrSortedValC,
                                    const int* csrSortedRowPtrC,
                                    const int* csrSortedColIndC,
                                    size_t* pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4 * sizeof(double) * 2;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseXcsrgeam2Nnz(alphasparseHandle_t handle,
                        int m,
                        int n,
                        const alphasparseMatDescr_t descrA,
                        int nnzA,
                        const int* csrSortedRowPtrA,
                        const int* csrSortedColIndA,
                        const alphasparseMatDescr_t descrB,
                        int nnzB,
                        const int* csrSortedRowPtrB,
                        const int* csrSortedColIndB,
                        const alphasparseMatDescr_t descrC,
                        int* csrSortedRowPtrC,
                        int* nnzTotalDevHostPtr,
                        void* workspace)
{
    const int threadPerBlock = 256;
    const int blockPerGrid   = min(32, (threadPerBlock + n - 1) / threadPerBlock);

    // printf("geam nnz \n");
    geam_nnz_per_row<<<blockPerGrid, threadPerBlock, 0, handle->stream>>> (m, n, csrSortedRowPtrA, csrSortedColIndA, csrSortedRowPtrB, csrSortedColIndB, csrSortedRowPtrC);

    cudaDeviceSynchronize();

    prefix<<< dim3(1), dim3(1), 0, handle->stream>>> (csrSortedRowPtrC, m + 1);

    cudaDeviceSynchronize();

    cudaMemcpy(nnzTotalDevHostPtr, csrSortedRowPtrC + m, sizeof(int), cudaMemcpyDeviceToHost);
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
alphasparseStatus_t
csrgeam2_template(alphasparseHandle_t handle,
                int m,
                int n,
                const void* alpha,
                const alphasparseMatDescr_t descrA,
                int nnzA,
                const void* csrSortedValA,
                const int* csrSortedRowPtrA,
                const int* csrSortedColIndA,
                const void* beta,
                const alphasparseMatDescr_t descrB,
                int nnzB,
                const void* csrSortedValB,
                const int* csrSortedRowPtrB,
                const int* csrSortedColIndB,
                const alphasparseMatDescr_t descrC,
                void* csrSortedValC,
                int* csrSortedRowPtrC,
                int* csrSortedColIndC,
                void* pBuffer)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL || descrB->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

    const int threadPerBlock = 256;
    const int blockPerGrid         = min(32, (threadPerBlock + n - 1) / threadPerBlock);

    add_plain<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>( m, *((T*)alpha), (T*)csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, *((T*)beta), (T*)csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (T*)csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (T*)pBuffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsrgeam2(alphasparseHandle_t handle,
                    int m,
                    int n,
                    const float* alpha,
                    const alphasparseMatDescr_t descrA,
                    int nnzA,
                    const float* csrSortedValA,
                    const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const float* beta,
                    const alphasparseMatDescr_t descrB,
                    int nnzB,
                    const float* csrSortedValB,
                    const int* csrSortedRowPtrB,
                    const int* csrSortedColIndB,
                    const alphasparseMatDescr_t descrC,
                    float* csrSortedValC,
                    int* csrSortedRowPtrC,
                    int* csrSortedColIndC,
                    void* pBuffer)
{ 
    csrgeam2_template<float>(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, 
                            descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
                            
    return ALPHA_SPARSE_STATUS_SUCCESS;
}                

alphasparseStatus_t
alphasparseDcsrgeam2(alphasparseHandle_t handle,
                    int m,
                    int n,
                    const double* alpha,
                    const alphasparseMatDescr_t descrA,
                    int nnzA,
                    const double* csrSortedValA,
                    const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const double* beta,
                    const alphasparseMatDescr_t descrB,
                    int nnzB,
                    const double* csrSortedValB,
                    const int* csrSortedRowPtrB,
                    const int* csrSortedColIndB,
                    const alphasparseMatDescr_t descrC,
                    double* csrSortedValC,
                    int* csrSortedRowPtrC,
                    int* csrSortedColIndC,
                    void* pBuffer)
{ 
    csrgeam2_template<double>(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, 
                            descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
                            
    return ALPHA_SPARSE_STATUS_SUCCESS;
}                

alphasparseStatus_t
alphasparseCcsrgeam2(alphasparseHandle_t handle,
                    int m,
                    int n,
                    const void* alpha,
                    const alphasparseMatDescr_t descrA,
                    int nnzA,
                    const void* csrSortedValA,
                    const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const void* beta,
                    const alphasparseMatDescr_t descrB,
                    int nnzB,
                    const void* csrSortedValB,
                    const int* csrSortedRowPtrB,
                    const int* csrSortedColIndB,
                    const alphasparseMatDescr_t descrC,
                    void* csrSortedValC,
                    int* csrSortedRowPtrC,
                    int* csrSortedColIndC,
                    void* pBuffer)
{ 
    csrgeam2_template<cuFloatComplex>(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, 
                            descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
                            
    return ALPHA_SPARSE_STATUS_SUCCESS;
}                

alphasparseStatus_t
alphasparseZcsrgeam2(alphasparseHandle_t handle,
                    int m,
                    int n,
                    const void* alpha,
                    const alphasparseMatDescr_t descrA,
                    int nnzA,
                    const void* csrSortedValA,
                    const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const void* beta,
                    const alphasparseMatDescr_t descrB,
                    int nnzB,
                    const void* csrSortedValB,
                    const int* csrSortedRowPtrB,
                    const int* csrSortedColIndB,
                    const alphasparseMatDescr_t descrC,
                    void* csrSortedValC,
                    int* csrSortedRowPtrC,
                    int* csrSortedColIndC,
                    void* pBuffer)
{ 
    csrgeam2_template<cuDoubleComplex>(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, 
                            descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
                            
    return ALPHA_SPARSE_STATUS_SUCCESS;
}                
