#include "alphasparse.h"
#include <iostream>
#include <cuda/std/complex>

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) ((dir == ALPHASPARSE_DIRECTION_ROW) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (block_dim * block_dim * (j) + (bi) * block_dim + (bj))
#define BSR_IND_C(j, bi, bj) (block_dim * block_dim * (j) + (bi) + (bj) * block_dim)

__global__ static void 
findDiag(int m, const int *bsrRowPtrA, const int *bsrColIndA, int *diag_pos)
{    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
        
    for (int i = tid; i < m; i += stride) {
        diag_pos[i] = -1;
        for (int j = bsrRowPtrA[i]; j < bsrRowPtrA[i+1]; j++) {
            if(bsrColIndA[j] == i){
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
bsrilu0_general_kernel  (int                     mb,
                        alphasparseDirection_t   dir,
                        const int* __restrict__  bsrRowPtrA,
                        const int* __restrict__  bsrColIndA,                        
                        T* __restrict__          bsrValA,
                        const int* __restrict__  bsr_diag_ind,
                        const int                block_dim,
                        int* __restrict__        done_array,
                        alphasparseIndexBase_t   idx_base,
                        int                      boost,
                        U                        boost_tol,
                        T                        boost_val)
{
    int lid = threadIdx.x & (WFSIZE - 1);
    int wid = threadIdx.x / WFSIZE;

    int idx = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if(idx >= mb)
    {
        return;
    }

    // Current row this wavefront is working on
    // int row = map[idx];
    for(int row = idx; row < mb; row+=mb)
    {
        // Diagonal entry point of the current row
        int row_diag = bsr_diag_ind[row];

        // Row entry point
        int row_begin = bsrRowPtrA[row] - idx_base;
        int row_end   = bsrRowPtrA[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(int j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                int bsr_col = bsrColIndA[j] - idx_base;

                // Process all lower matrix BSR blocks

                // Obtain corresponding row entry and exit point that corresponds with the
                // current BSR column. Actually, we skip all lower matrix column indices,
                // therefore starting with the diagonal entry.
                int diag_j    = bsr_diag_ind[bsr_col];
                int row_end_j = bsrRowPtrA[bsr_col + 1] - idx_base;

                // Check for structural pivot
                if(diag_j == -1)
                {
                    pivot = true;
                    break;
                }

                // Spin loop until dependency has been resolved
                int          local_done    = atomicOr(&done_array[bsr_col], 0);
                while(!local_done)
                {
                    local_done = atomicOr(&done_array[bsr_col], 0);
                }

                // Make sure dependencies are visible in global memory
                __threadfence();

                // Loop through all rows within the BSR block
                for(int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = bsrValA[BSR_IND(diag_j, bi, bi, dir)];

                    // Loop through all rows
                    for(int bk = lid; bk < block_dim; bk += WFSIZE)
                    {
                        T val = bsrValA[BSR_IND(j, bk, bi, dir)];

                        // This has already been checked for zero by previous computations
                        val /= diag;

                        // Update
                        bsrValA[BSR_IND(j, bk, bi, dir)] = val;

                        // Do linear combination

                        // Loop through all columns above the diagonal of the BSR block
                        for(int bj = bi + 1; bj < block_dim; ++bj)
                        {
                            T zero = {};
                            bsrValA[BSR_IND(j, bk, bj, dir)]
                                = (zero - val) * bsrValA[BSR_IND(diag_j, bi, bj, dir)] + bsrValA[BSR_IND(j, bk, bj, dir)];
                        }
                    }
                }

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(int k = diag_j + 1; k < row_end_j; ++k)
                {
                    int bsr_col_k = bsrColIndA[k] - idx_base;

                    // Search for matching column index in current row
                    int q         = row_begin + lid;
                    int bsr_col_j = (q < row_end) ? bsrColIndA[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsrColIndA[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int match = __ffsll(__ballot_sync(0xFFFFFFFF, bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        int m = __shfl_sync(0xFFFFFFFF, q, match - 1, WFSIZE);

                        for(int bi = lid; bi < block_dim; bi += WFSIZE)
                        {
                            for(int bj = 0; bj < block_dim; ++bj)
                            {
                                T sum = {};

                                for(int bk = 0; bk < block_dim; ++bk)
                                {
                                    sum = bsrValA[BSR_IND(j, bi, bk, dir)] * bsrValA[BSR_IND(k, bk, bj, dir)] + sum;
                                }

                                bsrValA[BSR_IND(m, bi, bj, dir)] -= sum;
                            }
                        }
                    }
                }
            }

            // Process diagonal
            if(bsrColIndA[row_diag] - idx_base == row)
            {
                for(int bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = bsrValA[BSR_IND(row_diag, bi, bi, dir)];

                    // Numeric boost
                    if(boost)
                    {
                        // diag = (boost_tol >= rocsparse_abs(diag)) ? boost_val : diag;

                        if(lid == 0)
                        {
                            bsrValA[BSR_IND(row_diag, bi, bi, dir)] = diag;
                        }
                    }
                    else
                    {
                        // Check for numeric pivot
                        if(is_zero(diag))
                        {
                            pivot = true;
                            continue;
                        }
                    }

                    for(int bk = bi + 1 + lid; bk < block_dim; bk += WFSIZE)
                    {
                        // Multiplication factor
                        T val = bsrValA[BSR_IND(row_diag, bk, bi, dir)];
                        val /= diag;

                        // Update
                        bsrValA[BSR_IND(row_diag, bk, bi, dir)] = val;

                        // Do linear combination
                        for(int bj = bi + 1; bj < block_dim; ++bj)
                        {
                            T zero = {};
                            bsrValA[BSR_IND(row_diag, bk, bj, dir)]
                                = (zero - val) * bsrValA[BSR_IND(row_diag, bi, bj, dir)] + bsrValA[BSR_IND(row_diag, bk, bj, dir)];
                        }
                    }
                }
            }

            // Process upper diagonal BSR blocks
            for(int j = row_diag + 1; j < row_end; ++j)
            {
                for(int bi = 0; bi < block_dim; ++bi)
                {
                    for(int bk = lid; bk < block_dim; bk += WFSIZE)
                    {
                        for(int bj = bi + 1; bj < block_dim; ++bj)
                        {
                            T zero = {};
                            bsrValA[BSR_IND(j, bj, bk, dir)]
                                = (zero - bsrValA[BSR_IND(row_diag, bj, bi, dir)]) * bsrValA[BSR_IND(j, bi, bk, dir)] + bsrValA[BSR_IND(j, bj, bk, dir)];
                        }
                    }
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        // Make sure updated bsrValA is written to global memory
        __threadfence();

        if(lid == 0)
        {
            // First lane writes "we are done" flag
            atomicOr(&done_array[row], 1);

            // if(pivot)
            // {
            //     // Atomically set minimum zero pivot, if found
            //     atomicMin(zero_pivot, row + idx_base);
            // }
        }
    }
}