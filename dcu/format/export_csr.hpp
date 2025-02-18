#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

template <typename TYPE>
alphasparseStatus_t export_csr(const alphasparse_matrix_t source,
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **rows_start,
                          ALPHA_INT **rows_end,
                          ALPHA_INT **col_indx,
                          TYPE **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    // check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_CSR, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    internal_spmat mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *rows_start = mat->row_data;
    *rows_end = *rows_start+1;
    *col_indx = mat->col_data;
    *values = (TYPE*)(mat->val_data);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}