#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "csrspgemm_device_op.h"

template <typename IndexType, typename DataType>
alphasparseStatus_t spgemm_csr_op(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const DataType alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const DataType beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    Meta meta;
    h_setup<IndexType>(matA, matB, matC, meta);
    h_symbolic_binning<IndexType>(matC, meta);
    h_symbolic<IndexType>(matA, matB, matC, meta);
    h_numeric_binning<IndexType>(matC, meta);
    if(matC->nnz != *meta.total_nnz)
    {
        matC->nnz = *meta.total_nnz;
        if(matC->val_data != nullptr) cudaFree(matC->val_data);
        if(matC->col_data != nullptr) cudaFree(matC->col_data);

        CHECK_CUDA(cudaMalloc(&matC->val_data, matC->nnz * sizeof(DataType)));
        CHECK_CUDA(cudaMalloc(&matC->col_data, matC->nnz * sizeof(IndexType)));
    }

    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, (int *)matC->row_data, (int *)matC->row_data, matC->rows + 1);
    h_numeric_full_occu<IndexType, DataType>(matA, matB, matC, meta, alpha);
    meta.release();
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
