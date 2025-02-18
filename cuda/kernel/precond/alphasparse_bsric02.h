#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <iostream>

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

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T>
__global__ static void 
bsric0_binsearch_kernel (int                     mb,
                        alphasparseDirection_t   dir,
                        const int* __restrict__  bsrRowPtrA,
                        const int* __restrict__  bsrColIndA,                        
                        T* __restrict__          bsrValA,
                        const int* __restrict__  bsr_diag_ind,
                        const int                block_dim,
                        int* __restrict__        done_array,
                        alphasparseIndexBase_t   idx_base)
{
    int lid = threadIdx.x & (WFSIZE - 1);
    int wid = threadIdx.x / WFSIZE;

    int idx = blockIdx.x + wid;

    // Current block row this wavefront is working on

    // int block_row = block_map[idx];
    for(int block_row = idx; block_row < mb; block_row += mb)
    {
        // Block diagonal entry point of the current block row
        int block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            __threadfence();

            if(lid == 0)
            {
                // Last lane in wavefront writes "we are done" flag for its block row
                atomicOr(&done_array[block_row], 1);
            }
            
            return;
        }

        // Block row entry point
        int block_row_begin = bsrRowPtrA[block_row] - idx_base;
        int block_row_end   = bsrRowPtrA[block_row + 1] - idx_base;

        for(int row = lid; row < block_dim; row += WFSIZE)
        {
            // Row sum accumulator
            T row_sum = {};

            // Loop over block columns of current block row
            for(int j = block_row_begin; j < block_row_diag; j++)
            {
                // Block column index currently being processes
                int block_col = bsrColIndA[j] - idx_base;

                // Beginning of the block row that corresponds to block_col
                int local_block_begin = bsrRowPtrA[block_col] - idx_base;

                // Block diagonal entry point of block row 'block_col'
                int local_block_diag = bsr_diag_ind[block_col];

                // Structural zero pivot, do not process this block row
                if(local_block_diag == -1)
                {
                    // If one thread in the warp breaks here, then all threads in
                    // the warp break so no divergence
                    break;
                }

                // Spin loop until dependency has been resolved
                int          local_done    = atomicOr(&done_array[block_col], 0);
                while(!local_done)
                {
                    local_done = atomicOr(&done_array[block_col], 0);
                }

                __threadfence();

                for(int k = 0; k < block_dim; k++)
                {
                    // Column index currently being processes
                    int col = block_dim * block_col + k;

                    // Load diagonal entry
                    T diag_val = bsrValA[block_dim * block_dim * local_block_diag + block_dim * k + k];

                    // Row has numerical zero pivot
                    if(is_zero(diag_val))
                    {
                        // Normally would break here but to avoid divergence set diag_val to one and continue
                        // The zero pivot has already been set so further computation does not matter
                        diag_val = {1.0f};
                    }

                    T val = {};

                    // Corresponding value
                    if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                    {
                        val = bsrValA[block_dim * block_dim * j + block_dim * row + k];
                    }
                    else
                    {
                        val = bsrValA[block_dim * block_dim * j + block_dim * k + row];
                    }

                    // Local row sum
                    T local_sum = {};

                    // Loop over the row the current column index depends on
                    // Each lane processes one entry
                    for(int p = local_block_begin; p < local_block_diag + 1; p++)
                    {
                        // Perform a binary search to find matching block columns
                        int l = block_row_begin;
                        int r = block_row_end - 1;
                        int m = (r + l) >> 1;

                        int block_col_j = bsrColIndA[m] - idx_base;
                        int block_col_p = bsrColIndA[p] - idx_base;

                        // Binary search for block column
                        while(l < r)
                        {
                            if(block_col_j < block_col_p)
                            {
                                l = m + 1;
                            }
                            else
                            {
                                r = m;
                            }

                            m           = (r + l) >> 1;
                            block_col_j = bsrColIndA[m] - idx_base;
                        }

                        // Check if a match has been found
                        if(block_col_j == block_col_p)
                        {
                            for(int q = 0; q < block_dim; q++)
                            {
                                if(block_dim * block_col_p + q < col)
                                {
                                    T vp = {};
                                    T vj = {};
                                    if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                                    {
                                        vp = bsrValA[block_dim * block_dim * p + block_dim * k + q];
                                        vj = bsrValA[block_dim * block_dim * m + block_dim * row + q];
                                    }
                                    else
                                    {
                                        vp = bsrValA[block_dim * block_dim * p + block_dim * q + k];
                                        vj = bsrValA[block_dim * block_dim * m + block_dim * q + row];
                                    }

                                    // If a match has been found, do linear combination
                                    local_sum = vp * conj(vj) + local_sum;
                                }
                            }
                        }
                    }

                    val     = (val - local_sum) / diag_val;
                    row_sum = val * conj(val) + row_sum;

                    if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                    {
                        bsrValA[block_dim * block_dim * j + block_dim * row + k] = val;
                    }
                    else
                    {
                        bsrValA[block_dim * block_dim * j + block_dim * k + row] = val;
                    }
                }
            }

            // Handle diagonal block column of block row
            for(int j = 0; j < block_dim; j++)
            {
                int row_diag = block_dim * block_dim * block_row_diag + block_dim * j + j;

                // Check if 'col' row is complete
                if(j == row)
                {
                    bsrValA[row_diag] = make_value<T>(alpha_sqrt(alpha_abs(bsrValA[row_diag] - row_sum)));
                }

                // Ensure previous writes to global memory are seen by all threads
                __threadfence();

                // Load diagonal entry
                T diag_val = bsrValA[row_diag];

                // Row has numerical zero pivot
                if(is_zero(diag_val))
                {
                    // Normally would break here but to avoid divergence set diag_val to one and continue
                    // The zero pivot has already been set so further computation does not matter
                    diag_val = {1.0f};
                }

                if(j < row)
                {
                    // Current value
                    T val = {};

                    // Corresponding value
                    if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                    {
                        val = bsrValA[block_dim * block_dim * block_row_diag + block_dim * row + j];
                    }
                    else
                    {
                        val = bsrValA[block_dim * block_dim * block_row_diag + block_dim * j + row];
                    }

                    // Local row sum
                    T local_sum = {};

                    T vk = {};
                    T vj = {};
                    for(int k = block_row_begin; k < block_row_diag; k++)
                    {
                        for(int q = 0; q < block_dim; q++)
                        {
                            if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                            {
                                vk = bsrValA[block_dim * block_dim * k + block_dim * j + q];
                                vj = bsrValA[block_dim * block_dim * k + block_dim * row + q];
                            }
                            else
                            {
                                vk = bsrValA[block_dim * block_dim * k + block_dim * q + j];
                                vj = bsrValA[block_dim * block_dim * k + block_dim * q + row];
                            }

                            // If a match has been found, do linear combination
                            local_sum = vk * conj(vj) + local_sum;
                        }
                    }

                    for(int q = 0; q < j; q++)
                    {
                        if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                        {
                            vk = bsrValA[block_dim * block_dim * block_row_diag + block_dim * j + q];
                            vj = bsrValA[block_dim * block_dim * block_row_diag + block_dim * row + q];
                        }
                        else
                        {
                            vk = bsrValA[block_dim * block_dim * block_row_diag + block_dim * q + j];
                            vj = bsrValA[block_dim * block_dim * block_row_diag + block_dim * q + row];
                        }

                        // If a match has been found, do linear combination
                        local_sum = vk * conj(vj) + local_sum;
                    }

                    val     = (val - local_sum) / diag_val;
                    row_sum = val * conj(val) + row_sum;

                    if(dir ==  ALPHASPARSE_DIRECTION_ROW)
                    {
                        bsrValA[block_dim * block_dim * block_row_diag + block_dim * row + j] = val;
                    }
                    else
                    {
                        bsrValA[block_dim * block_dim * block_row_diag + block_dim * j + row] = val;
                    }
                }

                __threadfence();
            }
        }

        __threadfence();

        if(lid == 0)
        {
            // Last lane writes "we are done" flag for current block row
            atomicOr(&done_array[block_row], 1);
        }
    }
}