#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "../../../../format/transpose_conj_dia.hpp"
#include "../../../../format/destroy_dia.hpp"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t gemm_dia_row_conj(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat conjugated_mat;
    transpose_conj_dia<TYPE>(mat, &conjugated_mat);
    alphasparseStatus_t status = gemm_dia_row(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_dia(conjugated_mat);
    return status;
}