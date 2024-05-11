#pragma once

#include "alphasparse.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>

alphasparseStatus_t
alphasparseXcoo2csr(const int* row_data, int nnz, int m, int* csrRowPtr)
{
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, row_data, nnz, m, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
