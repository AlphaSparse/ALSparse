#ifndef DESTROY_CSR_HPP
#define DESTROY_CSR_HPP
#include "alphasparse/format.h"
#include "alphasparse/util/malloc.h"
#include <alphasparse/util.h>

inline alphasparseStatus_t destroy_csr(internal_spmat A)
{
    alpha_free(A->row_data);
    alpha_free(A->col_data);
    alpha_free(A->val_data);
#ifdef __DCU__

    alpha_free_dcu(A->d_col_indx);
    alpha_free_dcu(A->d_row_ptr);
    alpha_free_dcu(A->d_values);
#endif

    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#endif