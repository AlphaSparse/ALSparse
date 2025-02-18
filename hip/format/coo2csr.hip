#pragma once

#include "alphasparse.h"
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>

alphasparseStatus_t
alphasparseXcoo2csr(const int* row_data, int nnz, int m, int* csrRowPtr)
{
    hipsparseHandle_t handle = NULL;
    hipsparseCreate(&handle);
    hipsparseXcoo2csr(handle, row_data, nnz, m, csrRowPtr, HIPSPARSE_INDEX_BASE_ZERO);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
