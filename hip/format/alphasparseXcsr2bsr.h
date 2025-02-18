#include "alphasparse.h"
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>

// alphasparseStatus_t
// alphasparseScsr2bsr(alphasparseHandle_t handle,
//                     alphasparseDirection_t dir,
//                     int m,
//                     int n,
//                     const alphasparseMatDescr_t descrA,
//                     const float* csrValA,
//                     const int* csrRowPtrA,
//                     const int* csrColIndA,
//                     int blockDim,
//                     const alphasparseMatDescr_t descrC,
//                     float* bsrValC,
//                     int* bsrRowPtrC,
//                     int* bsrColIndC)
// {
//     hipsparseHandle_t chandle = NULL;
//     hipsparseCreate(&chandle);
//     hipsparseScsr2bsr(chandle, dir, m, n,
//                     descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
//                     descrC, bsrValC, bsrRowPtrC, bsrColIndC);
//     return ALPHA_SPARSE_STATUS_SUCCESS;
// }

alphasparseStatus_t
alphasparseScsr2bsr(alphasparseHandle_t handle,
                    hipsparseDirection_t dir,
                    int m,
                    int n,
                    const hipsparseMatDescr_t descrA,
                    const float* csrValA,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    int blockDim,
                    const hipsparseMatDescr_t descrC,
                    float* bsrValC,
                    int* bsrRowPtrC,
                    int* bsrColIndC)
{
    hipsparseHandle_t chandle = NULL;
    hipsparseCreate(&chandle);
    hipsparseScsr2bsr(chandle, dir, m, n,
                    descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
                    descrC, bsrValC, bsrRowPtrC, bsrColIndC);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}