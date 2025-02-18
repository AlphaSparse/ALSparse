#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "../../../../format/transpose_coo.hpp"
#include "../../../../format/transpose_conj_coo.hpp"
#include "../../../../format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_coo_u_lo_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT r = 0; r < m; ++r)
        for (ALPHA_INT c = 0; c < n; c++)
        {
            J t;
            y[index2(r, c, ldy)] = alpha_mul(beta, y[index2(r, c, ldy)]);
            t = alpha_mul(alpha, x[index2(r, c, ldx)]);
            y[index2(r, c, ldy)] = alpha_add(y[index2(r, c, ldy)], t);
            // y[index2(r, c, ldy)] = y[index2(r, c, ldy)] * beta + x[index2(r, c, ldx)] * alpha;
        }

    for (ALPHA_INT ai = 0; ai < mat->nnz; ai++)
    {
        ALPHA_INT ac = mat->col_data[ai];
        ALPHA_INT r = mat->row_data[ai];
        if (ac < r)
        {
            // J val = alpha * ((J *)mat->val_data)[ai];
            J val;
            val = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
            for (ALPHA_INT c = 0; c < n; ++c)
                y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
                y[index2(ac, c, ldy)] = alpha_madde(y[index2(ac, c, ldy)], val, x[index2(r, c, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
