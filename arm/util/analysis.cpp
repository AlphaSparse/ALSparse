#include "alphasparse/util/analysis.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/opt.h"
#include <memory.h>
#include <assert.h>
#include <stdio.h>

void alphasparse_nnz_counter_coo(const ALPHA_INT *row_indx, const ALPHA_INT *col_indx, const ALPHA_INT nnz, ALPHA_INT *lo_p, ALPHA_INT *diag_p, ALPHA_INT *hi_p)
{
    ALPHA_INT thread_num = alpha_get_thread_num();
    ALPHA_INT *lo = alpha_malloc(sizeof(ALPHA_INT) * thread_num);
    ALPHA_INT *diag = alpha_malloc(sizeof(ALPHA_INT) * thread_num);
    ALPHA_INT *hi = alpha_malloc(sizeof(ALPHA_INT) * thread_num);
    memset(lo, '\0', sizeof(ALPHA_INT) * thread_num);
    memset(diag, '\0', sizeof(ALPHA_INT) * thread_num);
    memset(hi, '\0', sizeof(ALPHA_INT) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
    {
        ALPHA_INT thread_id = alpha_get_thread_id();
#ifdef _OPENMP
#pragma omp for
#endif
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            ALPHA_INT row = row_indx[i];
            ALPHA_INT col = col_indx[i];
            if (row < col)
            {
                hi[thread_id] += 1;
            }
            else if (row > col)
            {
                lo[thread_id] += 1;
            }
            else
            {
                diag[thread_id] += 1;
            }
        }
    }
    for (ALPHA_INT i = 1; i < thread_num; i++)
    {
        lo[0] += lo[i];
        diag[0] += diag[i];
        hi[0] += hi[i];
    }
    *lo_p = lo[0];
    *diag_p = diag[0];
    *hi_p = hi[0];

    alpha_free(lo);
    alpha_free(diag);
    alpha_free(hi);
}

ALPHA_INT alphasparse_nnz_compute(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_INT lo, const ALPHA_INT diag, const ALPHA_INT hi, const alphasparseOperation_t operation, const struct alpha_matrix_descr descr)
{
    ALPHA_INT nnz;
    if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
    {
        nnz = lo + diag + hi;
    }
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC || descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
    {
        if (descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
            nnz = lo;
        else if (descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
            nnz = hi;
        else
            assert(0);
        if (descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            nnz += diag;
        else if (descr.diag == ALPHA_SPARSE_DIAG_UNIT)
            nnz += rows;
        else
            assert(0);
    }
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
    {
        if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        {
            if (descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                nnz = lo;
            else if (descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                nnz = hi;
            else
                assert(0);
        }
        else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE || operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        {
            if (descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                nnz = hi;
            else if (descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                nnz = lo;
            else
                assert(0);
        }
        else
        {
            assert(0);
        }
        if (descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            nnz += diag;
        else if (descr.diag == ALPHA_SPARSE_DIAG_UNIT)
            nnz += rows;
        else
            assert(0);
    }
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
    {
        if (descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            nnz = diag;
        else if (descr.diag == ALPHA_SPARSE_DIAG_UNIT)
            nnz = rows;
        else
            assert(0);
    }
    else
    {
        assert(0);
    }
    return nnz;
}

ALPHA_INT64 alphasparse_operations_mm(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_INT lo, const ALPHA_INT diag, const ALPHA_INT hi, const alphasparseOperation_t operation, const struct alpha_matrix_descr descr, const ALPHA_INT columns, const alphasparse_datatype_t datatype)
{
    ALPHA_INT mul;
    ALPHA_INT add;
    ALPHA_INT madd;
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX || datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        mul = 7;
        add = 2;
        madd = 9;
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT || datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        mul = 1;
        add = 1;
        madd = 2;
    }
    else
    {
        assert(0);
    }
    ALPHA_INT64 nnz = alphasparse_nnz_compute(rows, cols, lo, diag, hi, operation, descr);
    ALPHA_INT64 op = (nnz * madd + rows * (2 * mul + add)) * columns;
    return op;
}

ALPHA_INT64 alphasparse_operations_mv(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_INT lo, const ALPHA_INT diag, const ALPHA_INT hi, const alphasparseOperation_t operation, const struct alpha_matrix_descr descr, const alphasparse_datatype_t datatype)
{
    return alphasparse_operations_mm(rows, cols, lo, diag, hi, operation, descr, 1, datatype);
}
