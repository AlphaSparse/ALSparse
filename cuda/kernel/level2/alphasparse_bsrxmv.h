#include "alphasparse.h"
#include "alphasparse/types.h" 

template <typename T>
__global__ static void
bsr_xmv_plain(alphasparseDirection_t layout,
               int sizeOfMask, 
               int mb,
               int nb,
               int nnzb,
               const T alpha,
               const T *bsrVal,
               const int* bsrMaskPtr,
               const int *bsrRowPtr,
               const int* bsrEndPtr,
               const int *bsrColInd,
               int bs,
               const T *x,
               const T beta,
               T *y)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int m_inner = mb;
    // int n_inner = nb;
   
    if (layout == ALPHASPARSE_DIRECTION_ROW) {
        for (int i = tid; i < sizeOfMask; i += stride) {
            int xi = bsrMaskPtr[i];
            for (int ai = bsrRowPtr[xi]; ai < bsrEndPtr[xi]; ai++) {   
                for (int row_inner = 0; row_inner < bs; row_inner++) {
                    y[bs * xi + row_inner] = y[bs * xi + row_inner] * beta;   
                    for (int col_inner = 0; col_inner < bs; col_inner++) {                                                                  
                        y[bs * xi + row_inner] += alpha * bsrVal[ai * bs * bs + row_inner * bs + col_inner] * x[bs * bsrColInd[ai] + col_inner];
                    }                    
                }
            }
        }
    }    
    else if (layout == ALPHASPARSE_DIRECTION_COLUMN) {
        for (int i = tid; i < sizeOfMask; i += stride) {
            int xi = bsrMaskPtr[i];
            for (int ai = bsrRowPtr[xi]; ai < bsrEndPtr[xi]; ai++) {                
                for (int col_inner = 0; col_inner < bs; col_inner++) {                    
                    for (int row_inner = 0; row_inner < bs; row_inner++) {   
                        if(!col_inner) y[bs * xi + row_inner] = y[bs * xi + row_inner] * beta; 
                        y[bs * xi + row_inner] += alpha * bsrVal[ai * bs * bs + col_inner * bs + row_inner] * x[bs * bsrColInd[ai] + col_inner];                      
                    }                    
                }
            }
        }
    }
}