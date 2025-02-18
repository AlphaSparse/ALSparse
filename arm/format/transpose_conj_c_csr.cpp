#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <alphasparse/compute.h>
#include <memory.h>

alphasparseStatus_t ONAME(const ALPHA_SPMAT_CSR *A, ALPHA_SPMAT_CSR **B)
{

    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    *B = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT colA = A->cols;
    mat->rows = colA;
    mat->cols = rowA;
    ALPHA_INT nnz = A->rows_end[rowA - 1];
    ALPHA_INT *rows_offset = alpha_memalign((uint64_t)(mat->rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->rows_start = rows_offset;
    mat->rows_end = rows_offset + 1;
    mat->col_indx = alpha_memalign((uint64_t)nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign((uint64_t)nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT col_counter[colA];
    ALPHA_INT row_offset[colA];
    memset(col_counter, '\0', (uint64_t)colA * sizeof(ALPHA_INT));
    for (ALPHA_INT i = 0; i < nnz; ++i)
    {
        col_counter[A->col_indx[i]] += 1;
    }
    row_offset[0] = 0;
    mat->rows_start[0] = 0;
    for (ALPHA_INT i = 1; i < colA; ++i)
    {
        row_offset[i] = row_offset[i - 1] + col_counter[i - 1];
        mat->rows_end[i - 1] = row_offset[i];
    }
    mat->rows_end[colA - 1] = nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->rows_end, mat->rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lrs = partition[tid];
        ALPHA_INT lrh = partition[tid + 1];
        for (ALPHA_INT r = 0; r < rowA; ++r)
        {
            for (ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ++ai)
            {
                ALPHA_INT ac = A->col_indx[ai];
                if (ac < lrs || ac >= lrh)
                    continue;
                ALPHA_INT index = row_offset[ac];
                mat->col_indx[index] = r;
                alpha_conj(mat->values[index], A->values[ai]);
                row_offset[ac] += 1;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
