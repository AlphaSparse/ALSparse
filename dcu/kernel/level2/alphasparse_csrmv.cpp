#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
#include <hip/hip_runtime.h>

// #include "alphasparse/spapi_dcu.h"
// #include "alphasparse/kernel_dcu.h"
#include "alphasparse/util/error.h"
#include "alphasparse/compute.h"
#include "alphasparse/opt.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/internal_check.h"
#include "./csrmv/dcu_gemv_csr.hpp"
#include "./csrmv/dcu_gemv_csr_trans.hpp"
#include "./csrmv/dcu_gemv_csr_conj.hpp"

template <typename TYPE>
alphasparseStatus_t
alphasparse_csrmv_template(alphasparseHandle_t handle,
      alphasparseOperation_t operation,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT nnz,
      const TYPE *alpha,
      const alpha_matrix_descr_t descr,
      const TYPE *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      alphasparse_mat_info_t info,
      const TYPE *x,
      const TYPE *beta,
      TYPE *y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    if (descr == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    // Check index base
    if (descr->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    // Check sizes
    if (m < 0 || n < 0 || nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (m == 0 || n == 0 || nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check the rest of pointer arguments
    if (csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || x == nullptr || y == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    TYPE l_alpha = *alpha;
    TYPE l_beta  = *beta;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)         
            return dcu_gemv_csr(handle, m, n, nnz, l_alpha, csr_val, csr_row_ptr, csr_col_ind, info, x, l_beta, y);
        else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)         
            return dcu_gemv_csr_trans(handle, m, n, nnz, l_alpha, csr_val, csr_row_ptr, csr_col_ind, info, x, l_beta, y);
        else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)         
            return dcu_gemv_csr_conj(handle, m, n, nnz, l_alpha, csr_val, csr_row_ptr, csr_col_ind, info, x, l_beta, y);
        else
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else {
        // doto
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define C_IMPL(ONAME, TYPE)                                 \
alphasparseStatus_t                                        \
ONAME(alphasparseHandle_t handle,                           \
      alphasparseOperation_t trans,                        \
      ALPHA_INT m,                                          \
      ALPHA_INT n,                                          \
      ALPHA_INT nnz,                                        \
      const TYPE *alpha,                                    \
      const alpha_matrix_descr_t descr,                     \
      const TYPE *csr_val,                                  \
      const ALPHA_INT *csr_row_ptr,                         \
      const ALPHA_INT *csr_col_ind,                         \
      alphasparse_mat_info_t info,                          \
      const TYPE *x,                                        \
      const TYPE *beta,                                     \
      TYPE *y)                                              \
{                                                           \
    return alphasparse_csrmv_template(handle,               \
                                      trans,                \
                                      m,                    \
                                      n,                    \
                                      nnz,                  \
                                      alpha,                \
                                      descr,                \
                                      csr_val,              \
                                      csr_row_ptr,          \
                                      csr_col_ind,          \
                                      info,                 \
                                      x,                    \
                                      beta,                 \
                                      y);                   \
}

C_IMPL(alphasparse_s_csrmv, float);
C_IMPL(alphasparse_d_csrmv, double);
C_IMPL(alphasparse_c_csrmv, ALPHA_Complex8);
C_IMPL(alphasparse_z_csrmv, ALPHA_Complex16);
#undef C_IMPL