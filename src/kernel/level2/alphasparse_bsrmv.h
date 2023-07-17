#include "alphasparse.h"

template <typename T>
__global__ static void
bsr_gemv_plain(alphasparseDirection_t layout,
               int mb,
               int nb,
               int nnzb,
               const T alpha,
               const T *bsrVal,
               const int *bsrRowPtr,
               const int *bsrColInd,
               int bs,
               const T *x,
               const T beta,
               T *y)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int m_inner = mb;
    // int n_inner = nb;
    
    for (int m = tid; m < m_inner; m += stride) {
        for (int i = 0; i < bs; i++) {
            y[m * bs + i] = y[m * bs + i] * beta;
            
        }
    }
    // T temp = {};
    
    if (layout == ALPHASPARSE_DIRECTION_ROW) {
        for (int i = tid; i < m_inner; i += stride) {
            for (int ai = bsrRowPtr[i]; ai < bsrRowPtr[i + 1]; ai++) {                
                for (int row_inner = 0; row_inner < bs; row_inner++) {
                    for (int col_inner = 0; col_inner < bs; col_inner++) {                        
                        y[bs * i + row_inner] += alpha * bsrVal[ai * bs * bs + row_inner * bs + col_inner] * x[bs * bsrColInd[ai] + col_inner];
                    }                    
                }
            }
        }
    }    
    else if (layout == ALPHASPARSE_DIRECTION_COLUMN) {
        for (int i = tid; i < m_inner; i += stride) {
            for (int ai = bsrRowPtr[i]; ai < bsrRowPtr[i + 1]; ai++) {                
                for (int col_inner = 0; col_inner < bs; col_inner++) {
                    for (int row_inner = 0; row_inner < bs; row_inner++) {
                        y[bs * i + row_inner] += alpha * bsrVal[ai * bs * bs + col_inner * bs + row_inner] * x[bs * bsrColInd[ai] + col_inner];                      
                    }                    
                }
            }
        }
    }
}