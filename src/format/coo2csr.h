#pragma once

#include "alphasparse.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>

template<typename T>
alphasparseStatus_t
alphasparseXcoo2csr(const int* row_data, int nnz, int m, T* csrRowPtr)
{
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseXcoo2csr(handle, row_data, nnz, m, (int*)csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
