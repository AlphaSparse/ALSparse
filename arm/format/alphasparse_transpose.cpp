/**
 * @brief implement for alphasparse_transpose intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/spapi.h"
#include "alphasparse/util.h"
#include "alphasparse/format.h"
#include "transpose_csr.hpp"

// alphasparseStatus_t transpose_datatype_coo(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return transpose_s_coo((const spmat_coo_s_t *)source, (spmat_coo_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return transpose_d_coo((const spmat_coo_d_t *)source, (spmat_coo_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return transpose_c_coo((const spmat_coo_c_t *)source, (spmat_coo_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return transpose_z_coo((const spmat_coo_z_t *)source, (spmat_coo_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

alphasparseStatus_t transpose_datatype_csr(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_csr<spmat_csr_s_t, float>((const spmat_csr_s_t *)source, (spmat_csr_s_t **)dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_csr<spmat_csr_d_t, double>((const spmat_csr_d_t *)source, (spmat_csr_d_t **)dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_csr<spmat_csr_c_t, ALPHA_Complex8>((const spmat_csr_c_t *)source, (spmat_csr_c_t **)dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_csr<spmat_csr_z_t, ALPHA_Complex16>((const spmat_csr_z_t *)source, (spmat_csr_z_t **)dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

// alphasparseStatus_t transpose_datatype_csc(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return transpose_s_csc((const spmat_csc_s_t *)source, (spmat_csc_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return transpose_d_csc((const spmat_csc_d_t *)source, (spmat_csc_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return transpose_c_csc((const spmat_csc_c_t *)source, (spmat_csc_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return transpose_z_csc((const spmat_csc_z_t *)source, (spmat_csc_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparseStatus_t transpose_datatype_bsr(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return transpose_s_bsr((const spmat_bsr_s_t *)source, (spmat_bsr_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return transpose_d_bsr((const spmat_bsr_d_t *)source, (spmat_bsr_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return transpose_c_bsr((const spmat_bsr_c_t *)source, (spmat_bsr_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return transpose_z_bsr((const spmat_bsr_z_t *)source, (spmat_bsr_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparseStatus_t transpose_datatype_sky(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return transpose_s_sky((const spmat_sky_s_t *)source, (spmat_sky_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return transpose_d_sky((const spmat_sky_d_t *)source, (spmat_sky_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return transpose_c_sky((const spmat_sky_c_t *)source, (spmat_sky_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return transpose_z_sky((const spmat_sky_z_t *)source, (spmat_sky_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparseStatus_t transpose_datatype_dia(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return transpose_s_dia((const spmat_dia_s_t *)source, (spmat_dia_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return transpose_d_dia((const spmat_dia_d_t *)source, (spmat_dia_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return transpose_c_dia((const spmat_dia_c_t *)source, (spmat_dia_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return transpose_z_dia((const spmat_dia_z_t *)source, (spmat_dia_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

alphasparseStatus_t transpose_datatype_format(const internal_spmat *source, internal_spmat **dest, alphasparse_datatype_t datatype, alphasparseFormat_t format)
{
    if (format == ALPHA_SPARSE_FORMAT_CSR)
    {
        return transpose_datatype_csr(source, dest, datatype);
    }
    // else if (format == ALPHA_SPARSE_FORMAT_COO)
    // {
    //     return transpose_datatype_coo(source, dest, datatype);
    // }
    // else if (format == ALPHA_SPARSE_FORMAT_CSC)
    // {
    //     return transpose_datatype_csc(source, dest, datatype);
    // }
    // else if (format == ALPHA_SPARSE_FORMAT_BSR)
    // {
    //     return transpose_datatype_bsr(source, dest, datatype);
    // }
    // // else if (format == ALPHA_SPARSE_FORMAT_SKY)
    // // {
    // //     return transpose_datatype_sky(source, dest, datatype);
    // // }
    // else if (format == ALPHA_SPARSE_FORMAT_DIA)
    // {
    //     return transpose_datatype_dia(source, dest, datatype);
    // }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t alphasparse_transpose(const alphasparse_matrix_t source, alphasparse_matrix_t *dest)
{
    check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    alphasparse_matrix *dest_ = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
    *dest = dest_;
    dest_->format = source->format;
    dest_->datatype_cpu = source->datatype_cpu;
    return transpose_datatype_format((const internal_spmat *)source->mat, (internal_spmat **)&dest_->mat, source->datatype_cpu, source->format);
}