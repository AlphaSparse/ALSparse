/**
 * @brief implement for alphasparse_?_mv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/inspector.h"
#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
// #include "alphasparse/tuning.h"
#include "alphasparse/util.h"
#include "./mv/gemv/gemv_csr.hpp"
#include <cstdio>

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t alphasparse_mv_template(  
    const alphasparseOperation_t op_rq,  // operation_request
    const J alpha, 
    const alphasparse_matrix_t A,
    const struct alpha_matrix_descr dscr_rq, 
    const J *x, const J beta, J *y) {
  check_null_return(A, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  
  check_return(!((A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)||
               (A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)||
               (A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)||
               (A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)),
               ALPHA_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
  if (op_rq == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
  // TODO use simplelist to record optimized history
  // alphasparse_matrix_t compute_mat = NULL;
  struct alpha_matrix_descr compute_descr = dscr_rq;
  alphasparseOperation_t compute_operation = op_rq;

  if (dscr_rq.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC ||
      dscr_rq.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
    // check if it is a square matrix
    check_return(A->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);
  // alphasparse_inspector *inspector = (alphasparse_inspector *)A->inspector;
  // if (inspector == NULL) {
  //   fprintf(stderr, "inspector not initialized!!\n");
  //   return ALPHA_SPARSE_STATUS_NOT_INITIALIZED;
  // }
  // alphasparse_inspector_mv *mv_inspector = inspector->mv_inspector;
  // detect if set_mv_hint hasn't been invoked yet, ignore it


  void * mat = A->mat;
  if (A->format == ALPHA_SPARSE_FORMAT_CSR) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_csr(alpha, A->mat->rows, A->mat->cols, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data, (J*)(A->mat->val_data), x, beta, y);
      else if(op_rq == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return gemv_csr_trans(alpha, A->mat->rows, A->mat->cols, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, beta, y);
      else
        return gemv_csr_conj(alpha, A->mat->rows, A->mat->cols, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, beta, y);
    } else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) {
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else {
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
  } 
  else {
    fprintf(stderr, "format not supported\n");
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
}

#define C_IMPL(ONAME, TYPE)                                                          \
    alphasparseStatus_t ONAME(                                                       \
        const alphasparseOperation_t op_rq,  /*operation_request*/                   \
        const TYPE alpha, const alphasparse_matrix_t A,                              \
        const struct alpha_matrix_descr  dscr_rq,                                    \
        /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +  alphasparse_diag_type_t */ \
        const TYPE *x, const TYPE beta, TYPE *y)                                     \
    {                                                                                \
        return alphasparse_mv_template(op_rq,                                        \
                                       alpha,                                        \
                                       A,                                            \
                                       dscr_rq,                                      \
                                       x,                                            \
                                       beta,                                         \
                                       y);                                           \
    }

C_IMPL(alphasparse_s_mv, float);
C_IMPL(alphasparse_d_mv, double);
C_IMPL(alphasparse_c_mv, ALPHA_Complex8);
C_IMPL(alphasparse_z_mv, ALPHA_Complex16);
#undef C_IMPL