#include <iostream>

#include "alphasparse.h"
#include "ns/BIN.hpp" 
#include "csrspgemm_device_ns.h"
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

template <typename T, typename U>
alphasparseStatus_t spgemm_csr_ns(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const U alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const U beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    cudaEvent_t event[2];
    float msec;
    for (int i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }

    BIN<T, BIN_NUM> bin(matA->rows);

    matC->rows = matA->rows;
    matC->cols = matB->cols;
    // c.device_malloc = true;
    if(matC->row_data == nullptr)
      cudaMalloc((void **)&(matC->row_data), sizeof(T) * (matC->rows + 1));
    
    bin.set_max_bin(matA->row_data, matA->col_data, matB->row_data, matA->rows, TS_S_P, TS_S_T);

    // cudaEventRecord(event[0], 0);
    bool alloc = hash_symbolic<T, U>(matA, matB, matC, bin);
    // cudaEventRecord(event[1], 0);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&msec, event[0], event[1]);
    // printf("HashSymbolic: %f ms\n", msec);
    if(alloc)
    {
      cudaMalloc((void **)&(matC->col_data), sizeof(T) * (matC->nnz));
      cudaMalloc((void **)&(matC->val_data), sizeof(U) * (matC->nnz));
    }    

    bin.set_min_bin(matA->rows, TS_N_P, TS_N_T);

    // cudaEventRecord(event[0], 0);
    hash_numeric<T, U, true>(matA, matB, matC, alpha, bin);
    // cudaEventRecord(event[1], 0);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&msec, event[0], event[1]);
    // printf("HashNumeric: %f ms\n", msec);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
