#include "alphasparse.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>

alphasparseStatus_t
alphasparseScsr2bsr(alphasparseHandle_t handle,
                    cusparseDirection_t dir,
                    int m,
                    int n,
                    const cusparseMatDescr_t descrA,
                    const float* csrValA,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    int blockDim,
                    const cusparseMatDescr_t descrC,
                    float* bsrValC,
                    int* bsrRowPtrC,
                    int* bsrColIndC)
{
    cusparseHandle_t chandle = NULL;
    cusparseCreate(&chandle);
    cusparseScsr2bsr(chandle, dir, m, n,
                    descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
                    descrC, bsrValC, bsrRowPtrC, bsrColIndC);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}