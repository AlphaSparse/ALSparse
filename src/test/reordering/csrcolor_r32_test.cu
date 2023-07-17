#include "../test_common.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseDirection_t dir_alpha;

int m, n, nnz;
int* csrRowPtr = NULL;
int *coo_row_index, *coo_col_index;
float* coo_values;

int cuda_ncolors;
int *cuda_coloring;
int *cuda_reordering;

int ict_ncolors;
int *ict_coloring;
int *ict_reordering;

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             cudaGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
             __LINE__,                                                         \
             cusparseGetErrorString(status),                                   \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

void cuda_color()
{
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Offload data to device
    int* d_csrRowPtr = NULL;
    int* dArow = NULL;
    int* d_csrColInd = NULL;
    float* d_csrVal = NULL;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrVal, sizeof(float) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (m + 1)));
    
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_csrColInd, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_csrVal, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, d_csrRowPtr);
    cusparseMatDescr_t descrA = 0;
    cusparseColorInfo_t info = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateColorInfo(&info));

    float fractionToColor = 0.5;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&cuda_coloring, sizeof(int) * (m + 1)));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&cuda_reordering, sizeof(int) * (m + 1)));

    CHECK_CUSPARSE(cusparseScsrcolor(handle, m, nnz, descrA, d_csrVal, d_csrRowPtr, d_csrColInd, &fractionToColor, &cuda_ncolors, cuda_coloring, cuda_reordering, info));
    cudaMemcpy(csrRowPtr, d_csrRowPtr, sizeof(int)* (m+1), cudaMemcpyDeviceToHost);
    // step 6: free resources
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyColorInfo(info);
    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrVal);
    cudaFree(d_csrColInd);
}

void alpha_color()
{
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    // Offload data to device
    int* d_csrRowPtr = NULL;
    int* dArow = NULL;
    int* d_csrColInd = NULL;
    float* d_csrVal = NULL;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrVal, sizeof(float) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (m + 1)));
    
    CHECK_CUDA(cudaMemcpy(
        dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_csrColInd, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_csrVal, coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, d_csrRowPtr);
    alphasparseMatDescr_t descrA = 0;
    alphasparseColorInfo_t info = ALPHA_SPARSE_OPAQUE;
    alphasparseCreateMatDescr(&descrA);

    float fractionToColor = 0.5;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&ict_coloring, sizeof(int) * (m + 1)));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&ict_reordering, sizeof(int) * (m + 1)));

    alphasparseScsrcolor(handle, m, nnz, descrA, d_csrVal, d_csrRowPtr, d_csrColInd, &fractionToColor, &ict_ncolors, ict_coloring, ict_reordering, info);
    // step 6: free resources
    PRINT_IF_CUDA_ERROR(cudaFree(d_csrRowPtr));
    PRINT_IF_CUDA_ERROR(cudaFree(d_csrVal));
    PRINT_IF_CUDA_ERROR(cudaFree(d_csrColInd));
}

int
main(int argc, const char* argv[])
{
    // args
    args_help(argc, argv);
    file = args_get_data_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);

    // read coo
    alpha_read_coo<float>(
        file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
    coo_order<int32_t, float>(nnz, coo_row_index, coo_col_index, coo_values);
    csrRowPtr = (int*)alpha_malloc(sizeof(int) * (m + 1));

    cuda_color();
    alpha_color();

    if(check_flag)
    {
        cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

        int *h_cuda_coloring = (int *)malloc(sizeof(int)*(m+1));
        int *h_cuda_reordering = (int *)malloc(sizeof(int)*(m+1));

        int *h_ict_coloring = (int *)malloc(sizeof(int)*(m+1));
        int *h_ict_reordering = (int *)malloc(sizeof(int)*(m+1));

        cudaMemcpy(h_cuda_coloring, cuda_coloring, sizeof(int)* (m+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cuda_reordering, cuda_reordering, sizeof(int)* (m+1), cudaMemcpyDeviceToHost);

        cudaMemcpy(h_ict_coloring, ict_coloring, sizeof(int)* (m+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ict_reordering, ict_reordering, sizeof(int)* (m+1), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        // CHECK CONSISTENCY: COUNT NUMBER OF COLORS IN HCOLORING
        // CHECK CONSISTENCY: CHECK ANY COLOR NOT BEING SHARED BY TWO ADJACENT NODES.
        // CHECK CUDA
        for(int i = 0; i < m; ++i)
        {
            auto icolor = h_cuda_coloring[i];
            for(int at = csrRowPtr[i]; at < csrRowPtr[i + 1]; ++at)
            {
                auto j = coo_col_index[at];
                if(i != j)
                {
                    auto jcolor = h_cuda_coloring[j];
                    status = (icolor != jcolor) ? CUSPARSE_STATUS_SUCCESS
                                                : CUSPARSE_STATUS_INTERNAL_ERROR;        
                    if(status != CUSPARSE_STATUS_SUCCESS) printf("something wrong on %d adjacent check!\n",i);
                }
            }
        }
        
        // Verification of the number of colors by counting them.
        // Check if colors are contiguous
        {
            int max_value = 0;
            for(int i = 0; i < m; ++i)
            {
                // Check value is well defined.
                status = (h_cuda_coloring[i] >= 0 && h_cuda_coloring[i] < m)
                                            ? CUSPARSE_STATUS_SUCCESS
                                            : CUSPARSE_STATUS_INTERNAL_ERROR;
                if(status != CUSPARSE_STATUS_SUCCESS) printf("cuda coloring %d is over limit %d!\n", i, h_cuda_coloring[i]);
                // Calculate maximum value.
                if(h_cuda_coloring[i] > max_value)
                {
                    max_value = h_cuda_coloring[i];
                }
            }
            ++max_value;

            bool* marker = new bool[max_value];
            for(int i = 0; i < max_value; ++i)
            {
                marker[i] = false;
            }

            for(int i = 0; i < m; ++i)
            {
                marker[h_cuda_coloring[i]] = true;
            }

            for(int i = 0; i < max_value; ++i)
            {
                status = marker[i] ? CUSPARSE_STATUS_SUCCESS
                                    : CUSPARSE_STATUS_INTERNAL_ERROR;
                if(status != CUSPARSE_STATUS_SUCCESS) printf("something wrong on %d contiguous check!\n", i);
            }               
            delete[] marker;
            // Compare the number of colors.
            if(max_value != cuda_ncolors) std::cout<<"ncolor is not matched"<<std::endl;
            else std::cout<<"ncolor is matched"<<std::endl;
        }

        if(cuda_reordering)
        {
            // Need to verify this is a valid permutation array..
            int * cache = (int *)malloc(sizeof(int)*m);
            for(int i = 0; i < m; ++i)
            {
                cache[i] = 0;
            }

            for(int i = 0; i < m; ++i)
            {
                status = (h_cuda_reordering[i] >= 0 && h_cuda_reordering[i] < m)
                                            ? CUSPARSE_STATUS_SUCCESS
                                            : CUSPARSE_STATUS_INTERNAL_ERROR;
                if(status != CUSPARSE_STATUS_SUCCESS) printf("cuda reordring %d is over limit %d!\n", i, h_cuda_coloring[i]);
                cache[h_cuda_reordering[i]] = 1;
            }

            for(int i = 0; i < m; ++i)
            {
                status = (cache[i] != 0) ? CUSPARSE_STATUS_SUCCESS
                                        : CUSPARSE_STATUS_INTERNAL_ERROR;      
                if(status != CUSPARSE_STATUS_SUCCESS) printf("something wrong on %d permutation check!\n", i);
            }
            
            free(cache);
        }
        printf("CUDA COLORING PASSED!!\n");
        // CHECK ALPHA
        for(int i = 0; i < m; ++i)
        {
            auto icolor = h_ict_coloring[i];
            for(int at = csrRowPtr[i]; at < csrRowPtr[i + 1]; ++at)
            {
                auto j = coo_col_index[at];
                if(i != j)
                {
                    auto jcolor = h_ict_coloring[j];
                    status = (icolor != jcolor) ? CUSPARSE_STATUS_SUCCESS
                                                : CUSPARSE_STATUS_INTERNAL_ERROR;      
                    if(status != CUSPARSE_STATUS_SUCCESS) printf("something wrong on %d adjacent check!\n", i);
                }
            }
        }
        
        // Verification of the number of colors by counting them.
        // Check if colors are contiguous
        {
            int max_value = 0;
            for(int i = 0; i < m; ++i)
            {
                // Check value is well defined.
                status = (h_ict_coloring[i] >= 0 && h_ict_coloring[i] < m)
                                            ? CUSPARSE_STATUS_SUCCESS
                                            : CUSPARSE_STATUS_INTERNAL_ERROR;
                if(status != CUSPARSE_STATUS_SUCCESS) printf("alpha coloring %d is over limit %d!\n", i, h_cuda_coloring[i]);
                // Calculate maximum value.
                if(h_ict_coloring[i] > max_value)
                {
                    max_value = h_ict_coloring[i];
                }
            }
            ++max_value;

            bool* marker = new bool[max_value];
            for(int i = 0; i < max_value; ++i)
            {
                marker[i] = false;
            }

            for(int i = 0; i < m; ++i)
            {
                marker[h_ict_coloring[i]] = true;
            }

            for(int i = 0; i < max_value; ++i)
            {
                status = marker[i] ? CUSPARSE_STATUS_SUCCESS
                                    : CUSPARSE_STATUS_INTERNAL_ERROR;      
                if(status != CUSPARSE_STATUS_SUCCESS) printf("something wrong on %d contiguous check!\n", i);
            }                
            delete[] marker;
            // Compare the number of colors.
            if(max_value != ict_ncolors) std::cout<<"ncolor is not matched"<<std::endl;
            else std::cout<<"ncolor is matched"<<std::endl;
        }

        if(ict_reordering)
        {
            // Need to verify this is a valid permutation array..
            int * cache = (int *)malloc(sizeof(int)*m);
            for(int i = 0; i < m; ++i)
            {
                cache[i] = 0;
            }

            for(int i = 0; i < m; ++i)
            {
                status = (h_ict_reordering[i] >= 0 && h_ict_reordering[i] < m)
                                            ? CUSPARSE_STATUS_SUCCESS
                                            : CUSPARSE_STATUS_INTERNAL_ERROR;
                if(status != CUSPARSE_STATUS_SUCCESS) printf("alpha reordring %d is over limit %d!\n", i, h_cuda_coloring[i]);
                cache[h_ict_reordering[i]] = 1;
            }

            for(int i = 0; i < m; ++i)
            {
                status = (cache[i] != 0) ? CUSPARSE_STATUS_SUCCESS
                                        : CUSPARSE_STATUS_INTERNAL_ERROR;   
                if(status != CUSPARSE_STATUS_SUCCESS) printf("something wrong on %d permutation check!\n", i);
            }
            free(cache);             
        }
        printf("ALPHA COLORING PASSED!!\n");
    }
}