#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <iostream>

__global__ static void 
findDiag(int m, const int *csrRowPtrA, const int *csrColIndA, int *diag_pos)
{    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
        
    for (int i = tid; i < m; i += stride) {
        diag_pos[i] = -1;
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++) {
            if(csrColIndA[j] == i){
                diag_pos[i] = j;
            }
        }
    }    
}

__global__ static void 
print_device(int m, const int *diag_pos)
{       
    for (int i = 0; i < m; i ++) {
        printf("%d pos : %d\n", i, diag_pos[i]);
    }    
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
__global__ static void 
csrilu0_binsearch_kernel(int m,
                        const int* __restrict__ csrRowPtrA,
                        const int* __restrict__ csrColIndA,
                        T* __restrict__ csrValA_valM,
                        const int* __restrict__ csr_diag_ind,
                        int* __restrict__ done,
                        alphasparseIndexBase_t idx_base,
                        int                  boost,
                        U                    boost_tol,
                        T                    boost_val)
{
    int lid = threadIdx.x & (WFSIZE - 1);
    int wid = threadIdx.x / WFSIZE;

    int idx = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= m)
    {
        return;
    }

    // Current row this wavefront is working on
    // int row = map[idx];
    for(int row = idx; row < m; row+=m)
    {
        // Diagonal entry point of the current row
        int row_diag = csr_diag_ind[row];

        // Row entry point
        int row_begin = csrRowPtrA[row] - idx_base;
        int row_end   = csrRowPtrA[row + 1] - idx_base;

        // Loop over column of current row
        for(int j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            int local_col = csrColIndA[j] - idx_base;

            // Corresponding value
            T local_val = csrValA_valM[j];

            // End of the row that corresponds to local_col
            int local_end = csrRowPtrA[local_col + 1] - idx_base;

            // Diagonal entry point of row local_col
            int local_diag = csr_diag_ind[local_col];

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = local_end - 1;
            }

            // Spin loop until dependency has been resolved
            int          local_done    = atomicOr(&done[local_col], 0);
            while(!local_done)
            {
                local_done = atomicOr(&done[local_col], 0);
            }

            // Make sure updated csrValA_valM is visible
            __threadfence();

            // Load diagonal entry
            T diag_val = csrValA_valM[local_diag];

            // Numeric boost
            if(boost)
            {
                // diag_val = (boost_tol >= alpha_abs(diag_val)) ? boost_val : diag_val;

                __threadfence();

                if(lid == 0)
                {
                    csrValA_valM[local_diag] = diag_val;
                }
            }
            else
            {
                // Row has numerical zero diagonal
                if(is_zero(diag_val))
                {
                    // Skip this row if it has a zero pivot
                    break;
                }
            }

            csrValA_valM[j] = local_val = local_val / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            int l = j + 1;
            for(int k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
            {
                // Perform a binary search to find matching columns
                int r     = row_end - 1;
                int m     = (r + l) >> 1;
                int col_j = csrColIndA[m];

                int col_k = csrColIndA[k];

                // Binary search
                while(l < r)
                {
                    if(col_j < col_k)
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m;
                    }

                    m     = (r + l) >> 1;
                    col_j = csrColIndA[m];
                }

                // Check if a match has been found
                if(col_j == col_k)
                {
                    // If a match has been found, do ILU computation
                    T zero = {};
                    csrValA_valM[l] = (zero-local_val) * csrValA_valM[k] + csrValA_valM[l];
                }
            }
        }

        // Make sure updated csrValA_valM is written to global memory
        __threadfence();

        if(lid == 0)
        {
            // First lane writes "we are done" flag
            atomicOr(&done[row], 1);
        }
    }
}