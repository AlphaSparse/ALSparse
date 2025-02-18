#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "../../../../format/transpose_bsr.hpp"
#include "../../../../format/transpose_conj_bsr.hpp"
#include "../../../../format/destroy_bsr.hpp"

template <typename J>
alphasparseStatus_t trmm_bsr_u_lo_col_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_bsr<J>(mat, &transposed_mat);
    alphasparseStatus_t status = trmm_bsr_u_hi_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_bsr(transposed_mat);
    return status;
}
