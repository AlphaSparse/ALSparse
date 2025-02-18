#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trmm_sky_n_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            y[index2(j, i, ldy)] = alpha_mul(y[index2(j, i, ldy)], beta);
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            ALPHA_INT start = mat->pointers[cr];
            ALPHA_INT end   = mat->pointers[cr + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end; ++ai)
            {
                ALPHA_INT ac = cr - eles_num + idx;
                if (ac <= cr)
                {
                    J t;
                    t = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                    y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], t, x[index2(cc, ac, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, ac, ldx)];
                }
                idx++;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
