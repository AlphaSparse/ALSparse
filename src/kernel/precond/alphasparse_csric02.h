#include "alphasparse.h"
#include <iostream>
#include <cuda/std/complex>

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

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__global__ static void 
csric0_binsearch_kernel(int m,
                        const int* __restrict__ csrRowPtrA,
                        const int* __restrict__ csrColIndA,
                        T* __restrict__ csrValA_valM,
                        const int* __restrict__ csr_diag_ind,
                        int* __restrict__ done,
                        // const int* __restrict__ map,
                        // int* __restrict__ zero_pivot,
                        alphasparseIndexBase_t idx_base)
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
    for(int row = idx; row < m; row+=m)
    {
        // int row = map[idx];
        // Diagonal entry point of the current row
        int row_diag = csr_diag_ind[row];

        // Row entry point
        int row_begin = csrRowPtrA[row] - idx_base;
        int row_end   = csrRowPtrA[row + 1] - idx_base;

        // Row sum accumulator
        T sum = {};
        // Loop over column of current row
        for(int j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            int local_col = csrColIndA[j] - idx_base;

            // Corresponding value
            T local_val = csrValA_valM[j];

            // Beginning of the row that corresponds to local_col
            int local_begin = csrRowPtrA[local_col] - idx_base;

            // Diagonal entry point of row local_col
            int local_diag = csr_diag_ind[local_col];
           
            // Local row sum
            T local_sum = {};

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }
            // printf("row %d local_begin %d local diag %d row diag %d col %d\n", row, local_begin, local_diag, row_diag, local_col);
            // Spin loop until dependency has been resolved
            int          local_done    = atomicOr(&done[local_col], 0);
            // unsigned int times_through = 0;
            while(!local_done)
            {
                // local_done = rocsparse_atomic_load(&done[local_col], __ATOMIC_ACQUIRE);
                local_done = atomicOr(&done[local_col], 0);
                // if(row == local_col && row == 0) local_done = 1;
            }

            // Make sure updated csrValA_valM is visible globally
            __threadfence();

            // Load diagonal entry
            T diag_val = csrValA_valM[local_diag];

            // Row has numerical zero diagonal
            if(is_zero(diag_val))
            {
                // Skip this row if it has a zero pivot
                break;
            }

            // Compute reciprocal
            // diag_val = 1.0f / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            int l = row_begin;
            for(int k = local_begin + lid; k < local_diag; k += WFSIZE)
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
                    // If a match has been found, do linear combination
                    local_sum = csrValA_valM[k] * conj(csrValA_valM[m]) + local_sum;
                }
            }

            // Accumulate row sum
            // printf("row %d lid %d local sum %f\n", row, lid, local_sum);
            local_sum = alpha_reduce_sum<WFSIZE>(local_sum);
            // Last lane id computes the Cholesky factor and writes it to global memory
            // printf("AF lid %d local sum %f\n", lid, local_sum);
            if(lid == 0)
            {
                // printf("row %d lid %d local sum %f local val %f diag %f\n", row, lid, local_sum, local_val, diag_val);
                local_val = (local_val - local_sum) / diag_val;
                sum       = local_val * conj(local_val) + sum;

                csrValA_valM[j] = local_val;                
            }
        }
    
        if(lid == 0)
        {
            // Last lane processes the diagonal entry
            if(row_diag >= 0)
            {               
                csrValA_valM[row_diag] = make_value<T>(alpha_sqrt(alpha_abs(csrValA_valM[row_diag] - sum)));
                // printf("row %d lid %d row_diag %d sum %f csrValA_valM[row_diag] %f \n", row, lid, row_diag, sum, csrValA_valM[row_diag]);
            }
        }

        // Make sure csrValA_valM is written to global memory
        __threadfence();

        if(lid == 0)
        {
            // Last lane writes "we are done" flag
            atomicOr(&done[row], 1);
        }
    }
}