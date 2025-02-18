#include "alphasparse.h"
#include "alphasparse/types.h" 

template <typename T>
__global__ static void
bsrsv2_u_hi_plain(alphasparseDirection_t dir,
                    int mb,
                    int nnzb,
                    const T alpha,
                    const T* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int bs,
                    alpha_bsrsv2Info_t info,
                    const T* x,
                    T* y,
                    T* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int block_rowA = mb;
    // int rowA = mb * bs;  
    for (int r = block_rowA - 1; r >=0 ; r--)
    {
        for (int ai = bsrRowPtrA[r + 1]-1; ai >= bsrRowPtrA[r]; ai--)
        {
            int ac = bsrColIndA[ai];
            if(ac == r) //对角块
            {
                for(int block_r = bs-1; block_r>=0 ; block_r--)
                {
                    for(int block_c = bs-1; block_c >= 0; block_c--) //块内上三角
                    {
			            if(block_r == block_c)
                        {
                            y[r*bs + block_r] = (alpha * x[r*bs + block_r] - pBuffer[r*bs + block_r]);
                            continue;
                        }
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            // printf("here\n");
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
            else if (ac > r) //上三角块
            {
                for(int block_r = 0; block_r < bs; block_r++)
                {
                    for(int block_c = 0; block_c < bs; block_c++)
                    {
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__global__ static void
bsrsv2_u_lo_plain(alphasparseDirection_t dir,
                    int mb,
                    int nnzb,
                    const T alpha,
                    const T* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int bs,
                    alpha_bsrsv2Info_t info,
                    const T* x,
                    T* y,
                    T* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int block_rowA = mb;

    for (int r = 0; r < block_rowA; r++)
    {
        for (int ai = bsrRowPtrA[r]; ai < bsrRowPtrA[r + 1]; ai++)
        {
            int ac = bsrColIndA[ai];
            if(ac == r) //对角块
            {
                for(int block_r = 0; block_r < bs; block_r++)
                {
                    for(int block_c = 0; block_c <= block_r; block_c++) //块内下三角
                    {
			            if(block_r == block_c)
                        {
                            y[r*bs + block_r] = (alpha * x[r*bs + block_r] - pBuffer[r*bs + block_r]);
                            continue;
                        }
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
            else if (ac < r) //下三角块
            {
                for(int block_r = 0; block_r < bs; block_r++)
                {
                    for(int block_c = 0; block_c < bs; block_c++)
                    {
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__global__ static void
bsrsv2_n_hi_plain(alphasparseDirection_t dir,
                    int mb,
                    int nnzb,
                    const T alpha,
                    const T* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int bs,
                    alpha_bsrsv2Info_t info,
                    const T* x,
                    T* y,
                    T* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int block_rowA = mb;
    T * diag = &pBuffer[mb * bs];

    for (int ar = 0; ar < block_rowA; ++ar)
    {
        for (int ai = bsrRowPtrA[ar]; ai < bsrRowPtrA[ar + 1]; ++ai)
        {
            if (bsrColIndA[ai] == ar) //对角块
            {
                for(int block_i = 0; block_i < bs; block_i++) //访问块内对角元素
                {
                    diag[ar * bs + block_i] = bsrValA[ai * bs * bs + block_i * bs + block_i];
                }
            } 
        }   
    }

    for (int r = block_rowA - 1; r >=0 ; r--)
    {
        for (int ai = bsrRowPtrA[r + 1]-1; ai >= bsrRowPtrA[r]; ai--)
        {
            int ac = bsrColIndA[ai];
            if(ac == r) //对角块
            {
                for(int block_r = bs-1; block_r>=0 ; block_r--)
                {
                    for(int block_c = bs-1; block_c >= 0; block_c--) //块内上三角
                    {
			            if(block_r == block_c)
                        {
                            y[r*bs + block_r] = (alpha * x[r*bs + block_r] - pBuffer[r*bs + block_r]) / diag[r*bs + block_r];
                            continue;
                        }
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            // printf("here\n");
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
            else if (ac > r) //上三角块
            {
                for(int block_r = 0; block_r < bs; block_r++)
                {
                    for(int block_c = 0; block_c < bs; block_c++)
                    {
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__global__ static void
bsrsv2_n_lo_plain(alphasparseDirection_t dir,
                    int mb,
                    int nnzb,
                    const T alpha,
                    const T* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int bs,
                    alpha_bsrsv2Info_t info,
                    const T* x,
                    T* y,
                    T* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int block_rowA = mb;
    T * diag = &pBuffer[mb * bs];

    for (int ar = 0; ar < block_rowA; ++ar)
    {
        for (int ai = bsrRowPtrA[ar]; ai < bsrRowPtrA[ar + 1]; ++ai)
        {
            if (bsrColIndA[ai] == ar) //对角块
            {
                for(int block_i = 0; block_i < bs; block_i++) //访问块内对角元素
                {
                    diag[ar * bs + block_i] = bsrValA[ai * bs * bs + block_i * bs + block_i];
                }
            } 
        }   
    }

    for (int r = 0; r < block_rowA; r++)
    {
        for (int ai = bsrRowPtrA[r]; ai < bsrRowPtrA[r + 1]; ai++)
        {
            int ac = bsrColIndA[ai];
            if(ac == r) //对角块
            {
                for(int block_r = 0; block_r < bs; block_r++)
                {
                    for(int block_c = 0; block_c <= block_r; block_c++) //块内下三角
                    {
			            if(block_r == block_c)
                        {
                            y[r*bs + block_r] = (alpha * x[r*bs + block_r] - pBuffer[r*bs + block_r]) / diag[r*bs + block_r];
                            continue;
                        }
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
            else if (ac < r) //下三角块
            {
                for(int block_r = 0; block_r < bs; block_r++)
                {
                    for(int block_c = 0; block_c < bs; block_c++)
                    {
                        if(dir == ALPHASPARSE_DIRECTION_ROW)
                        {
                            // A row major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                        }
                        else
                        {
                            // A column major
                            pBuffer[r*bs + block_r] += bsrValA[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                        }
                    }
                }
            }
        }
    }
}