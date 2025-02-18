#pragma once

#include "alphasparse.h"
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include "transpose_csr.h"
#include "alphasparse_create_csc.h"

template<typename T, typename U>
alphasparseStatus_t
alphasparseCsr2csc(alphasparseSpMatDescr_t &csr, alphasparseSpMatDescr_t &csc)
{
    transpose_csr<T, U>(csr);
    alphasparseCreateCsc(&csc,
                        csr->cols,
                        csr->rows,
                        csr->nnz,
                        csr->row_data,
                        csr->col_data,
                        csr->val_data,
                        csr->row_type,
                        csr->col_type,
                        csr->idx_base,
                        csr->data_type);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
