#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <assert.h>
#include <string.h>

#include "./bsrmv/bsrmv_kernels.h"

template <typename TYPE>
static alphasparseStatus_t gemv_gebsr_for_each_thread(
    const TYPE alpha, const internal_spmat A, const TYPE *x,
    const TYPE beta, TYPE *y, ALPHA_INT lrs, ALPHA_INT lre) {
  ALPHA_INT rbd = A->row_block_dim;
  ALPHA_INT cbd = A->col_block_dim;
  ALPHA_INT m_inner = A->rows;
  ALPHA_INT n_inner = A->cols;
  ALPHA_INT task_rows = (lre - lrs) * rbd;
  TYPE *tmp = alpha_malloc(sizeof(TYPE) * task_rows);
  memset(tmp, 0, sizeof(TYPE) * task_rows);

#ifdef S
  if (rbd == 4 && cbd == 1) {
    __spmv_bsr4x1_serial_host_sse_float(alpha, beta, lre - lrs,
                                        A->row_data + lrs, A->col_data,
                                        ((TYPE *)A->val_data), x, y + lrs * 4);
  } else {
    fprintf(stderr, " gebsr float not supported\n");
    return ALPHA_SPARSE_STATUS_EXECUTION_FAILED;
  }
#else
  if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
    for (ALPHA_INT i = lrs, j = 0; i < lre; i++, j++) {
      for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++) {
        TYPE *val = &((TYPE *)A->val_data)[ai * cbd * rbd];
        const TYPE *rhs = &x[cbd * A->col_data[ai]];
        for (ALPHA_INT row_inner = 0; row_inner < rbd; row_inner++) {
          for (ALPHA_INT col_inner = 0; col_inner < cbd; col_inner++) {
            tmp[rbd * j + row_inner] = alpha_madde(tmp[rbd * j + row_inner],
                        val[row_inner * cbd + col_inner],
                        x[cbd * A->col_data[ai] + col_inner]);
          }
        }
      }
    }
  }
  // For Fortran, block_layout is defaulted as col_major
  else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    for (ALPHA_INT i = lrs, j = 0; i < lre; i++, j++) {
      for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++) {
        TYPE *res = &tmp[rbd * j];
        TYPE *val = &((TYPE *)A->val_data)[ai * cbd * rbd];
        const TYPE *rhs = &x[cbd * A->col_data[ai]];
        for (ALPHA_INT col_inner = 0; col_inner < cbd; col_inner++) {
          for (ALPHA_INT row_inner = 0; row_inner < rbd; row_inner++) {
            res[row_inner] = alpha_madde(res[row_inner], val[col_inner * rbd + row_inner],
                        x[cbd * A->col_data[ai] + col_inner]);
          }
        }
      }
    }
  } else
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;

  for (ALPHA_INT m = lrs * rbd, m_t = 0; m < lre * rbd; m++, m_t++) {
    y[m] = alpha_mul(y[m], beta);
    y[m] = alpha_madde(y[m], tmp[m_t], alpha);
  }
  free(tmp);
#endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_gebsr(const TYPE alpha, const internal_spmat A,
                           const TYPE *x, const TYPE beta,
                           TYPE *y) {
  ALPHA_INT m_inner = A->rows;
  ALPHA_INT thread_num = alpha_get_thread_num();

  ALPHA_INT partition[thread_num + 1];
  balanced_partition_row_by_nnz(A->rows_end, m_inner, thread_num, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT local_m_s = partition[tid];
    ALPHA_INT local_m_e = partition[tid + 1];
    gemv_gebsr_for_each_thread(alpha, A, x, beta, y, local_m_s, local_m_e);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
