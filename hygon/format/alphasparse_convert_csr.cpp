#include "alphasparse/format.h"
#include "alphasparse/inspector.h"
#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/internal_check.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/types.h"
#include "convert_csr_coo.hpp"

alphasparseStatus_t convert_csr_datatype_coo(const internal_spmat source,
                                             internal_spmat *dest,
                                             alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_csr_coo<ALPHA_INT, float, _internal_spmat>(source, dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_csr_coo<ALPHA_INT, double, _internal_spmat>(source, dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    return convert_csr_coo<ALPHA_INT, ALPHA_Complex8, _internal_spmat>(source, dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    return convert_csr_coo<ALPHA_INT, ALPHA_Complex16, _internal_spmat>(source, dest);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

// alphasparseStatus_t convert_csr_datatype_csr(const internal_spmat *source, internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_csr_s_csr((spmat_csr_s_t *)source, (spmat_csr_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_csr_d_csr((spmat_csr_d_t *)source, (spmat_csr_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_csr_c_csr((spmat_csr_c_t *)source, (spmat_csr_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_csr_z_csr((spmat_csr_z_t *)source, (spmat_csr_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparseStatus_t convert_csr_datatype_csc(const internal_spmat *source, internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_csr_s_csc((spmat_csc_s_t *)source, (spmat_csr_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_csr_d_csc((spmat_csc_d_t *)source, (spmat_csr_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_csr_c_csc((spmat_csc_c_t *)source, (spmat_csr_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_csr_z_csc((spmat_csc_z_t *)source, (spmat_csr_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparseStatus_t convert_csr_datatype_bsr(const internal_spmat *source, internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_csr_s_bsr((spmat_bsr_s_t *)source, (spmat_csr_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_csr_d_bsr((spmat_bsr_d_t *)source, (spmat_csr_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_csr_c_bsr((spmat_bsr_c_t *)source, (spmat_csr_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_csr_z_bsr((spmat_bsr_z_t *)source, (spmat_csr_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

alphasparseStatus_t convert_csr_datatype_format(const internal_spmat source,
                                                internal_spmat *dest,
                                                alphasparse_datatype_t datatype,
                                                alphasparseFormat_t format) {
  if (format == ALPHA_SPARSE_FORMAT_COO) {
    return convert_csr_datatype_coo(source, dest, datatype);
  }
  // else if (format == ALPHA_SPARSE_FORMAT_CSR)
  // {
  //     return convert_csr_datatype_csr(source, dest, datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_CSC)
  // {
  //     return convert_csr_datatype_csc(source, dest, datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_BSR)
  // {
  //     return convert_csr_datatype_bsr(source, dest, datatype);
  // }
  else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparseStatus_t alphasparse_convert_csr(
    const alphasparse_matrix_t source,       /* convert original matrix to CSR representation */
    const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  if (source->format != ALPHA_SPARSE_FORMAT_COO) {
    *dest = NULL;
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  alphasparse_matrix *dest_ = (alphasparse_matrix *)alpha_malloc(sizeof(alphasparse_matrix));
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_CSR;
  dest_->datatype_cpu = source->datatype_cpu;
  dest_->inspector = NULL;
  dest_->inspector = NULL;
  dest_->inspector = (alphasparse_inspector_t)alpha_malloc(sizeof(alphasparse_inspector));
  alphasparse_inspector *kernel_inspector = (alphasparse_inspector *)dest_->inspector;
  kernel_inspector->mv_inspector = NULL;
  kernel_inspector->request_kernel = ALPHA_NONE;
  kernel_inspector->mm_inspector = NULL;
  kernel_inspector->mmd_inspector = NULL;
  kernel_inspector->sv_inspector = NULL;
  kernel_inspector->sm_inspector = NULL;
  kernel_inspector->memory_policy = ALPHA_SPARSE_MEMORY_AGGRESSIVE;
  alphasparseStatus_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_csr_datatype_format((const internal_spmat )source->mat,
                                       (internal_spmat *)&dest_->mat, source->datatype_cpu,
                                       source->format);
  } else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    alphasparse_matrix_t AA;
    check_error_return(alphasparse_transpose(source, &AA));
    status =
        convert_csr_datatype_format((const internal_spmat )AA->mat,
                                    (internal_spmat *)&dest_->mat, AA->datatype_cpu, AA->format);
    alphasparse_destroy(AA);
    return status;
  } else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparseStatus_t alphasparse_convert_csr_internal(
    const alphasparse_matrix_t source,       /* convert original matrix to BSR representation */
    const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  return alphasparse_convert_csr(source, operation, dest);
}