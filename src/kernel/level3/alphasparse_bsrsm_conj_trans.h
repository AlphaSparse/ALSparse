#include "alphasparse.h"

//float and double do nothing
template <typename T>
__global__ static void
bsrsm2_n_up_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const T alpha,
                const T* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const T* B,
                int ldb,
                T* X,
                int ldx,
                T* pBuffer)
{
    
}

template <typename T>
__global__ static void
bsrsm2_u_up_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const T alpha,
                const T* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const T* B,
                int ldb,
                T* X,
                int ldx,
                T* pBuffer)
{
    
}

template <typename T>
__global__ static void
bsrsm2_n_lo_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const T alpha,
                const T* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const T* B,
                int ldb,
                T* X,
                int ldx,
                T* pBuffer)
{
    
}

template <typename T>
__global__ static void
bsrsm2_u_lo_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const T alpha,
                const T* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const T* B,
                int ldb,
                T* X,
                int ldx,
                T* pBuffer)
{
    
}

//float complex
__global__ static void
bsrsm2_n_up_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuFloatComplex alpha,
                const cuFloatComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuFloatComplex* B,
                int ldb,
                cuFloatComplex* X,
                int ldx,
                cuFloatComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    cuFloatComplex* diag = pBuffer;
    const int m = mb * bs;
    const int bs2 = bs * bs;

    for(int br = 0 ; br < mb; br++){
        for(int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++){
            int bc = bsrSortedColInd[ai];
            if(bc == br){
                for(int b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(br,b_row,bs)] = bsrSortedVal[ai * bs2 +  b_row *(bs + 1)];
                }
            }
        }
    }

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + m + tid * bs;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //right-bottom most
                for(int row = bs - 1; row >=0 ; row--)
                {
                    //upper triangle of block
                    for(int col = row + 1 ; col < bs ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConjf(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = (alpha * cuConjf(B[(br * bs + row) * ldb + out_x_col]) - temp[row]) / diag[row + br * bs];
                }
            }
        }
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + m + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //right-bottom most
                for(int col = bs - 1; col >= 0; col--)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = (alpha * cuConjf(B[B0_offset + (br * bs + col) * ldb]) - temp[col]) / diag[col + br * bs];
                  
                    for(int row = col - 1; row >= 0; row--){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }        
        }
    }
}

__global__ static void
bsrsm2_u_up_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuFloatComplex alpha,
                const cuFloatComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuFloatComplex* B,
                int ldb,
                cuFloatComplex* X,
                int ldx,
                cuFloatComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int m = mb * bs;
    const int bs2 = bs * bs;

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + tid * bs;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //right-bottom most
                for(int row = bs - 1; row >=0 ; row--)
                {
                    //upper triangle of block
                    for(int col = row + 1 ; col < bs ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConjf(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = alpha * cuConjf(B[(br * bs + row) * ldb + out_x_col]) - temp[row];
                }
            }
        }
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //right-bottom most
                for(int col = bs - 1; col >= 0; col--)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = alpha * cuConjf(B[B0_offset + (br * bs + col) * ldb]) - temp[col];
                  
                    for(int row = col - 1; row >= 0; row--){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }        
        }
    }
}

__global__ static void
bsrsm2_n_lo_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuFloatComplex alpha,
                const cuFloatComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuFloatComplex* B,
                int ldb,
                cuFloatComplex* X,
                int ldx,
                cuFloatComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    cuFloatComplex* diag = pBuffer;
    const int m = mb * bs;
    const int bs2 = bs * bs;

    for(int br = 0 ; br < mb; br++){
        for(int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++){
            int bc = bsrSortedColInd[ai];
            if(bc == br){
                for(int b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(br,b_row,bs)] = bsrSortedVal[ai * bs2 +  b_row *(bs + 1)];
                }
            }
        }
    }

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + m + tid * bs;
            for (int br = 0 ; br < mb ; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                    {
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //left-top most
                for(int row = 0; row < bs ; row++)
                {
                    //upper triangle of block
                    for(int col = 0 ; col < row ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        // int X_offset =  (br * bs + col)  + out_x_col * ldx;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConjf(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = (alpha * cuConjf(B[(br * bs + row) * ldb + out_x_col]) - temp[row]) / diag[row + br * bs];
                }
            }
        }  
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + m + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = 0; br < mb; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx ;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //top-left most
                for(int col = 0; col < bs; col++)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = (alpha * cuConjf(B[B0_offset + (br * bs + col) * ldb]) - temp[col]) / diag[col + br * bs];

                    for(int row = col + 1; row < bs; row++){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }
        }
    }
}

__global__ static void
bsrsm2_u_lo_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuFloatComplex alpha,
                const cuFloatComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuFloatComplex* B,
                int ldb,
                cuFloatComplex* X,
                int ldx,
                cuFloatComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int m = mb * bs;
    const int bs2 = bs * bs;

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + tid * bs;
            for (int br = 0 ; br < mb ; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                    {
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //left-top most
                for(int row = 0; row < bs ; row++)
                {
                    //upper triangle of block
                    for(int col = 0 ; col < row ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        // int X_offset =  (br * bs + col)  + out_x_col * ldx;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConjf(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = alpha * cuConjf(B[(br * bs + row) * ldb + out_x_col]) - temp[row];
                }
            }
        }  
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuFloatComplex* temp = pBuffer + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = 0; br < mb; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx ;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConjf(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //top-left most
                for(int col = 0; col < bs; col++)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = cuConjf(alpha * B[B0_offset + (br * bs + col) * ldb]) - temp[col];

                    for(int row = col + 1; row < bs; row++){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }
        }
    }
}

//double complex
__global__ static void
bsrsm2_n_up_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuDoubleComplex alpha,
                const cuDoubleComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuDoubleComplex* B,
                int ldb,
                cuDoubleComplex* X,
                int ldx,
                cuDoubleComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    cuDoubleComplex* diag = pBuffer;
    const int m = mb * bs;
    const int bs2 = bs * bs;

    for(int br = 0 ; br < mb; br++){
        for(int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++){
            int bc = bsrSortedColInd[ai];
            if(bc == br){
                for(int b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(br,b_row,bs)] = bsrSortedVal[ai * bs2 +  b_row *(bs + 1)];
                }
            }
        }
    }

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + m + tid * bs;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //right-bottom most
                for(int row = bs - 1; row >=0 ; row--)
                {
                    //upper triangle of block
                    for(int col = row + 1 ; col < bs ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConj(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = (alpha * cuConj(B[(br * bs + row) * ldb + out_x_col]) - temp[row]) / diag[row + br * bs];
                }
            }
        }
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + m + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //right-bottom most
                for(int col = bs - 1; col >= 0; col--)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = (alpha * cuConj(B[B0_offset + (br * bs + col) * ldb]) - temp[col]) / diag[col + br * bs];
                  
                    for(int row = col - 1; row >= 0; row--){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }        
        }
    }
}

template <typename T>
__global__ static void
bsrsm2_u_up_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuDoubleComplex alpha,
                const cuDoubleComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuDoubleComplex* B,
                int ldb,
                cuDoubleComplex* X,
                int ldx,
                cuDoubleComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int m = mb * bs;
    const int bs2 = bs * bs;

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + tid * bs;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //right-bottom most
                for(int row = bs - 1; row >=0 ; row--)
                {
                    //upper triangle of block
                    for(int col = row + 1 ; col < bs ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConj(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = alpha * cuConj(B[(br * bs + row) * ldb + out_x_col]) - temp[row];
                }
            }
        }
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = mb - 1; br >= 0; br--)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc > br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //right-bottom most
                for(int col = bs - 1; col >= 0; col--)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = alpha * cuConj(B[B0_offset + (br * bs + col) * ldb]) - temp[col];
                  
                    for(int row = col - 1; row >= 0; row--){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }        
        }
    }
}

template <typename T>
__global__ static void
bsrsm2_n_lo_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuDoubleComplex alpha,
                const cuDoubleComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuDoubleComplex* B,
                int ldb,
                cuDoubleComplex* X,
                int ldx,
                cuDoubleComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    cuDoubleComplex* diag = pBuffer;
    const int m = mb * bs;
    const int bs2 = bs * bs;

    for(int br = 0 ; br < mb; br++){
        for(int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br + 1]; ai++){
            int bc = bsrSortedColInd[ai];
            if(bc == br){
                for(int b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(br,b_row,bs)] = bsrSortedVal[ai * bs2 +  b_row *(bs + 1)];
                }
            }
        }
    }

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + m + tid * bs;
            for (int br = 0 ; br < mb ; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                    {
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //left-top most
                for(int row = 0; row < bs ; row++)
                {
                    //upper triangle of block
                    for(int col = 0 ; col < row ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        // int X_offset =  (br * bs + col)  + out_x_col * ldx;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConj(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = (alpha * cuConj(B[(br * bs + row) * ldb + out_x_col]) - temp[row]) / diag[row + br * bs];
                }
            }
        }  
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + m + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = 0; br < mb; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx ;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //top-left most
                for(int col = 0; col < bs; col++)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = (alpha * cuConj(B[B0_offset + (br * bs + col) * ldb]) - temp[col]) / diag[col + br * bs];

                    for(int row = col + 1; row < bs; row++){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }
        }
    }
}

__global__ static void
bsrsm2_u_lo_conj_trans_plain(alphasparseDirection_t dirA,
                int mb,
                int n,
                int nnzb,
                const cuDoubleComplex alpha,
                const cuDoubleComplex* bsrSortedVal,
                const int* bsrSortedRowPtr,
                const int* bsrSortedColInd,
                int bs,
                const cuDoubleComplex* B,
                int ldb,
                cuDoubleComplex* X,
                int ldx,
                cuDoubleComplex* pBuffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int m = mb * bs;
    const int bs2 = bs * bs;

    if(dirA == ALPHASPARSE_DIRECTION_ROW)
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + tid * bs;
            for (int br = 0 ; br < mb ; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                    {
                        //row-major
                        for(int row = 0; row < bs; row++)
                        {
                        //all entities belongs to upper triangle 
                            int a0_offset = ai * bs2 +  row * bs;
                            for(int col = 0 ; col < bs ; col++)
                            {
                                int X_offset =  (bc * bs + col) * ldx + out_x_col;
                                int ele_offset =  a0_offset + col;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //row-major
                //left-top most
                for(int row = 0; row < bs ; row++)
                {
                    //upper triangle of block
                    for(int col = 0 ; col < row ; col++){
                        int X_offset =  (br * bs + col) * ldx + out_x_col;
                        // int X_offset =  (br * bs + col)  + out_x_col * ldx;
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  row * bs + col] * cuConj(X[X_offset]);
                    }
                    X[(br * bs + row) * ldx + out_x_col] = alpha * cuConj(B[(br * bs + row) * ldb + out_x_col]) - temp[row];
                }
            }
        }  
    }
    else
    {
        for(int out_x_col = tid; out_x_col < n; out_x_col+=stride)
        {
            cuDoubleComplex* temp = pBuffer + tid * bs;
            const int X0_offset = out_x_col;
            const int B0_offset = out_x_col;

            for (int br = 0; br < mb; br++)
            {
                for(int i = 0 ; i < bs ; i++){
                    temp[i] = {};
                }
                int diagBlock = -1;
                // memset(temp,'\0', bs * sizeof(ALPHA_Number));
                for (int ai = bsrSortedRowPtr[br]; ai < bsrSortedRowPtr[br+1]; ai++)
                {
                    int bc = bsrSortedColInd[ai];
                    if(bc < br)
                        //col-major
                        for(int col = 0; col < bs; col++)
                        {
                        //all entities belongs to upper triangle 
                            int X_offset =  X0_offset + (bc * bs + col) * ldx ;
                            int a0_offset = ai * bs2 +  col * bs;
                            for(int row = 0 ; row < bs ; row++)
                            {
                                int ele_offset =  a0_offset + row;
                                temp[row] += bsrSortedVal[ ele_offset ] * cuConj(X[X_offset]);
                            }
                        }
                    //diagonal must be none-zero block
                    if( bc==br ){
                        diagBlock = ai;
                    }
                }
                if(diagBlock == -1)
                {
                    printf("lhs matrix invalid for trsm!!!\n");
                }
                //col-major
                //top-left most
                for(int col = 0; col < bs; col++)
                {
                    //upper triangle of block
                    X[X0_offset + (br * bs + col) * ldx] = alpha * cuConj(B[B0_offset + (br * bs + col) * ldb]) - temp[col];

                    for(int row = col + 1; row < bs; row++){
                        temp[row] += bsrSortedVal[ diagBlock * bs2 +  col * bs + row] * X[X0_offset + (br * bs + col) * ldx];
                    }
                }
            }
        }
    }
}