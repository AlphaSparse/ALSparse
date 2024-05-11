#ifndef __Z_SETUP_CUH__
#define __Z_SETUP_CUH__

#include "define.h"

template <typename IndexType = int>
__global__ void __launch_bounds__(1024, 2) k_compute_flop(
    const IndexType* __restrict__ d_arpt, 
    const IndexType* __restrict__ d_acol,
    const IndexType* __restrict__ d_brpt,
    IndexType M,
    IndexType *d_row_flop,
    IndexType *d_max_row_flop){

    __shared__ IndexType shared_max_row_flop[1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    if(threadIdx.x == 0){
        shared_max_row_flop[0] = 0;
    }
    __syncthreads();
    IndexType row_flop = 0;
    IndexType j;
    IndexType acol;
    IndexType arow_start, arow_end;
    arow_start = d_arpt[i];
    arow_end = d_arpt[i+1];
    for (j = arow_start; j < arow_end; j++) {
        acol = d_acol[j];
        row_flop += d_brpt[acol + 1] - d_brpt[acol];
    }
    d_row_flop[i] = row_flop;
    atomicMax(shared_max_row_flop, row_flop);
    __syncthreads();
    if(threadIdx.x == 0){
        atomicMax(d_max_row_flop, shared_max_row_flop[0]);
    }
}


#endif
