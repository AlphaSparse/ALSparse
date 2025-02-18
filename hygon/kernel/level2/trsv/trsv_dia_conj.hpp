#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

#include "../../../format/transpose_dia.hpp"
#include "../../../format/transpose_conj_dia.hpp"
#include "../../../format/destroy_dia.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_dia_n_hi_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_dia<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_dia_n_lo(alpha, conjugated_mat, x, y);
    destroy_dia(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_n_lo_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_dia<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_dia_n_hi(alpha, conjugated_mat, x, y);
    destroy_dia(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_u_hi_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_dia<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_dia_u_lo(alpha, conjugated_mat, x, y);
    destroy_dia(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_u_lo_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_dia<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_dia_u_hi(alpha, conjugated_mat, x, y);
    destroy_dia(conjugated_mat);
    return status;
}