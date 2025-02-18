#include "./ellmv/ellmv_kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#include <string.h>

alphasparseStatus_t gemv_ell_row_partiton(
    const float alpha, const internal_spmat A, const ALPHA_INT lrs,
    const ALPHA_INT lre, const float *x, const float beta,
    float *y) {
  const ALPHA_INT m = A->rows;
  const ALPHA_INT width = A->ell_width;
  const ALPHA_INT lrl = lre - lrs;

  __spmv_ell_serial_host_sse_float(alpha, beta, lrs, lre, m, width, A->ind_data,
                                   ((float *)A->val_data), x, y);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_ell_row_partiton(
    const TYPE alpha, const internal_spmat A, const ALPHA_INT lrs,
    const ALPHA_INT lre, const TYPE *x, const TYPE beta,
    TYPE *y) {
  const ALPHA_INT m = A->rows;
  const ALPHA_INT width = A->ell_width;
  const ALPHA_INT lrl = lre - lrs;

  for (ALPHA_INT r = lrs; r < lre; r++) {
    TYPE tmp;
    tmp = alpha_setzero(tmp);
    for (ALPHA_INT c = 0; c < width; c++) {
      TYPE val = ((TYPE *)A->val_data)[c * m + r];
      ALPHA_INT idx = A->ind_data[c * m + r];
      tmp = alpha_madd(val, x[idx], tmp);
    }
    y[r] = alpha_mul(y[r], beta);
    y[r] = alpha_madd(tmp, alpha, y[r]);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_ell_omp(const TYPE alpha,
                                         const internal_spmat mat,
                                         const TYPE *x,
                                         const TYPE beta,
                                         TYPE *y) {
  const ALPHA_INT rows = mat->rows;
  const ALPHA_INT cols = mat->cols;
  const ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    const ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT r_squeeze = (rows + NPERCL - 1) / NPERCL;
    ALPHA_INT lrs = (ALPHA_INT64)tid * r_squeeze / thread_num;
    ALPHA_INT lre = (ALPHA_INT64)(tid + 1) * r_squeeze / thread_num;
    lrs *= NPERCL;
    lre *= NPERCL;
    lrs = alpha_min(lrs, rows);
    lre = alpha_min(lre, rows);
    gemv_ell_row_partiton(alpha, mat, lrs, lre, x, beta, y);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
template <typename TYPE>
alphasparseStatus_t gemv_ell(const TYPE alpha, const internal_spmat mat,
                           const TYPE *x, const TYPE beta,
                           TYPE *y) {
  return gemv_ell_omp(alpha, mat, x, beta, y);
}
