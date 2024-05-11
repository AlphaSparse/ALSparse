#pragma once

#include "alphasparse.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include "transpose_csc.h"
#include "alphasparse_create_csr.h"

template<typename T, typename U>
alphasparseStatus_t
alphasparseCsc2csr(alphasparseSpMatDescr_t &csc, alphasparseSpMatDescr_t &csr)
{
    transpose_csc<T, U>(csc);
    alphasparseCreateCsr(&csr,
                        csc->cols,
                        csc->rows,
                        csc->nnz,
                        csc->col_data,
                        csc->row_data,
                        csc->val_data,
                        csc->col_type,
                        csc->row_type,
                        csc->idx_base,
                        csc->data_type);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
