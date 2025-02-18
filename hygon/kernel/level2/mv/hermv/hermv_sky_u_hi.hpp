#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
hermv_sky_u_hi(const TYPE alpha,
	  const internal_spmat A,
	  const TYPE *x,
	  const TYPE beta,
	  TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		y[i] = alpha_mul(y[i], beta);
		y[i] = alpha_madde(y[i], alpha, x[i]);
		// y[i] = beta * y[i] + alpha * x[i];
	}

	for(ALPHA_INT c = 0; c < n; ++c)
    {
		const ALPHA_INT col_start = A->pointers[c];
		const ALPHA_INT col_end = A->pointers[c + 1];
		ALPHA_INT col_indx = 1;
		for(ALPHA_INT i = col_start; i < col_end; i++)
		{
			ALPHA_INT col_eles = col_end - col_start;
			TYPE v;
			TYPE v_c;
			v_c = cmp_conj(((TYPE *)A->val_data)[i]);
			v_c = alpha_mul(v_c, alpha);
			v = ((TYPE *)A->val_data)[i];
			if(i != col_end - 1)
			{
				ALPHA_INT r = c - col_eles + col_indx;
				v = alpha_mul(v, alpha);
				y[r] = alpha_madde(y[r], v, x[c]);
				y[c] = alpha_madde(y[c], v_c, x[r]);
				col_indx ++;
			}
		}
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
