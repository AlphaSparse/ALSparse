#include "alphasparse.h"
#include "alphasparse/types.h" 

template <typename T>
__global__ static void
bsrmm_plain(alphasparseDirection_t dirA,
              int mb,
              int n,
              int kb,
              int nnzb,
              const T alpha,
              const alphasparseMatDescr_t descrA,
              const T* bsrValA,
              const int* bsrRowPtrA,
              const int* bsrColIndA,
              int bs,
              const T* B,
              int ldb,
              const T beta,
              T* C,
              int ldc)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int m = mb * bs;

    for (int j = tid * bs; j < n; j += stride * bs) {
        for (int i = 0; i < m; ++i) {
            for (int l = 0; l < bs; l++) {
                C[index2(j + l, i, ldc)] *= beta;
            }
        }
    }

    switch (dirA)
    {
        case ALPHASPARSE_DIRECTION_ROW:
            for (int c = tid * bs; c < n; c += bs * stride) { // choose a column from x
                for (int r = 0; r < m; r += bs) { // choose a block of row
                    int br = r / bs;
                    for (int ai = bsrRowPtrA[br]; ai < bsrRowPtrA[br + 1]; ++ai) { // choose a block
                        const T *blk = &bsrValA[ai * bs * bs];
                        for (int cc = 0; cc < bs; ++cc)
                        {
                            for (int lr = 0; lr < bs; ++lr) 
                            { // choose a inner row
                                int ac = bsrColIndA[ai] * bs;
                                for (int lc = 0; lc < bs; ++lc) {
                                    C[index2(c + cc, r + lr, ldc)] += alpha * blk[index2(lr, lc, bs)] * B[index2(c + cc, ac + lc, ldb)];
                                }
                            }
                        }
                    }
                }
            }
            break;

        case ALPHASPARSE_DIRECTION_COLUMN:
            for (int c = tid * bs; c < n; c += bs * stride) { // choose a column from x
                for (int r = 0; r < m; r += bs) { // choose a block of row
                    int br = r / bs;
                    for (int ai = bsrRowPtrA[br]; ai < bsrRowPtrA[br + 1]; ++ai)
                    { // choose a block
                        for (int cc = 0; cc < bs; ++cc)
                        {
                            for (int lr = 0; lr < bs; ++lr) { // choose a inner row
                                int ac       = bsrColIndA[ai] * bs;
                                const T *blk = &bsrValA[ai * bs * bs];
                                for (int lc = 0; lc < bs; ++lc) {
                                    C[index2(c + cc, r + lr, ldc)] += alpha * blk[index2(lc, lr, bs)] * B[index2(c + cc, ac + lc, ldb)];
                                }
                            }
                        }
                    }
                }
            }
            break;
    }
}

template <typename T>
__global__ static void
bsrmm_trans_plain(alphasparseDirection_t dirA,
              int mb,
              int n,
              int kb,
              int nnzb,
              const T alpha,
              const alphasparseMatDescr_t descrA,
              const T* bsrValA,
              const int* bsrRowPtrA,
              const int* bsrColIndA,
              int bs,
              const T* B,
              int ldb,
              const T beta,
              T* C,
              int ldc)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int m      = mb * bs;
    
    for (int j = tid * bs; j < n; j += stride * bs) {
        for (int i = 0; i < m; ++i) {
            for (int l = 0; l < bs; l++) {
                C[index2(j + l, i, ldc)] *= beta;
            }
        }
    }

    switch (dirA) {
        case ALPHASPARSE_DIRECTION_ROW:
            for (int c = tid * bs; c < n; c += bs * stride) { // choose a column from x
                for (int r = 0; r < m; r += bs) { // choose a block of row
                    int br = r / bs;
                    for (int ai = bsrRowPtrA[br]; ai < bsrRowPtrA[br + 1]; ++ai) { // choose a block
                        const T *blk = &bsrValA[ai * bs * bs];
                        for (int cc = 0; cc < bs; ++cc)
                            for (int lr = 0; lr < bs; ++lr) { // choose a inner row
                                int ac = bsrColIndA[ai] * bs;

                                for (int lc = 0; lc < bs; ++lc) {
                                    C[index2(c + cc, r + lr, ldc)] += alpha * blk[index2(lr, lc, bs)] * B[index2(ac + lc, c + cc, ldb)];
                                }
                            }
                    }
                }
            }
            break;

        case ALPHASPARSE_DIRECTION_COLUMN:
            for (int c = tid * bs; c < n; c += bs * stride) { // choose a column from x
                for (int r = 0; r < m; r += bs) { // choose a block of row
                    int br = r / bs;
                    for (int ai = bsrRowPtrA[br]; ai < bsrRowPtrA[br + 1]; ++ai) { // choose a block
                        for (int cc = 0; cc < bs; ++cc)
                            for (int lr = 0; lr < bs; ++lr) { // choose a inner row

                                int ac       = bsrColIndA[ai] * bs;
                                const T *blk = &bsrValA[ai * bs * bs];

                                for (int lc = 0; lc < bs; ++lc) {
                                    C[index2(c + cc, r + lr, ldc)] += alpha * blk[index2(lc, lr, bs)] * B[index2(ac + lc, c + cc, ldb)];
                                }
                            }
                    }
                }
            }
            break;
    }
}