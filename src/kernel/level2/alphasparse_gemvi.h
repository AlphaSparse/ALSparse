#include "alphasparse.h"

template <typename T>
__global__ static void
gemvi_plain(int m,
            int n,
            const T alpha,
            const T* A,
            int lda,
            int nnz,
            const T* x,
            const int* xInd,
            const T beta,
            T* y,
            alphasparseIndexBase_t idxBase,
            void* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int ia = 0; ia < m; ia++)
    {
        y[ia] = beta * y[ia];
        for(int i = 0; i < nnz; i++)
        {
            int xina = xInd[i];
            y[ia] += alpha * A[ia + lda * xina] * x[i];//column major for cusparse
        }
    }    
}

template <typename T>
__global__ static void
gemvi_trans_plain(int m,
            int n,
            const T alpha,
            const T* A,
            int lda,
            int nnz,
            const T* x,
            const int* xInd,
            const T beta,
            T* y,
            alphasparseIndexBase_t idxBase,
            void* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int ia = 0; ia < m; ia++)
    {
        y[ia] = beta * y[ia];
        for(int i = 0; i < nnz; i++)
        {
            int xina = xInd[i];
            y[ia] += alpha * A[ia * lda + xina] * x[i];//row major for cusparse trans
        }    
    }   
}