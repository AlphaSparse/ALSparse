#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "../../../../format/transpose_coo.hpp"
#include "../../../../format/transpose_conj_coo.hpp"
#include "../../../../format/destroy_coo.hpp"
#include <stdio.h>
template <typename J>
alphasparseStatus_t symm_coo_n_hi_col_conj(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat conjugated_mat;
    transpose_conj_coo<J>(mat, &conjugated_mat);
    alphasparseStatus_t status = symm_coo_n_lo_col(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(conjugated_mat);
    return status;
}
