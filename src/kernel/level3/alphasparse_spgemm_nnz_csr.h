#include "alphasparse.h"
#include <cub/cub.cuh>  
#include "csrspgemm_device.h"

// struct CustomMax{    
//     template <typename T>    __device__ __forceinline__    
//     T operator()(const T &a, const T &b) const {        
//         return (b < a) ? a : b;    
//         }
//     };

template<typename T>
alphasparseStatus_t
spgemm_nnz_csr(alphasparseHandle_t handle,
               T m,
               T n,
               T k,
               T nnz_A,
               const T* csr_row_ptr_A,
               const T* csr_col_ind_A,
               T nnz_B,
               const T* csr_row_ptr_B,
               const T* csr_col_ind_B,
               T nnz_D,
               const T* csr_row_ptr_D,
               const T* csr_col_ind_D,
               T* csr_row_ptr_C,
               T* nnz_C,
               void * externalBuffer2)
{
    char* buffer = reinterpret_cast<char*>(externalBuffer2);

    size_t cudaPrim_size;
    void*  cudaPrim_buffer;

    // Compute number of intermediate products for each row
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
    csrgemm_intermediate_products<CSRGEMM_DIM, CSRGEMM_SUB><<<
                       dim3((m - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                       dim3(CSRGEMM_DIM)>>>
                       (m,
                       csr_row_ptr_A,
                       csr_col_ind_A,
                       csr_row_ptr_B,
                       csr_row_ptr_D,
                       csr_row_ptr_C,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       true,
                       false);
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    
    T * int_max_d;
    cudaMalloc((void **)& int_max_d, sizeof(T));   
    // Determine maximum of all intermediate products
    cub::DeviceReduce::Max(nullptr,
                              cudaPrim_size,
                              csr_row_ptr_C,
                              int_max_d,
                              m);
    cudaPrim_buffer = reinterpret_cast<void*>(buffer);

    cub::DeviceReduce::Max(cudaPrim_buffer,
                              cudaPrim_size,
                              csr_row_ptr_C,
                              int_max_d,
                              m);
    
    // CustomMax    min_op;
    // cub::DeviceReduce::Reduce(nullptr, cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C + m, m, min_op, 0);
    // // Allocate temporary storage
    // cudaPrim_buffer = reinterpret_cast<void*>(buffer);
    // // Run reduction
    // cub::DeviceReduce::Reduce(cudaPrim_buffer, cudaPrim_size, csr_row_ptr_C,  csr_row_ptr_C + m, m, min_op, 0);

    T int_max;    
    
    cudaMemcpy(&int_max, int_max_d, sizeof(T), cudaMemcpyDeviceToHost);
    // cudaMemcpy(csr_row_ptr_C + m, int_max_d, sizeof(T), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&int_max, csr_row_ptr_C + m, sizeof(T), cudaMemcpyDeviceToHost);
    // T *total;
    // total = (T *)malloc(sizeof(T)*(m+1)); 
    // cudaMemcpy(total, csr_row_ptr_C, (m+1) * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("init\n");
    //     for(int i = 0; i < m; i ++)
    //     {
    //         printf("i %d csr_row_ptr_C %d\n",i, total[i]);
    //     }
    // printf("int max %d buffer_size %d\n",int_max, int_max1);
    // Group offset buffer
    T* d_group_offset = reinterpret_cast<T*>(buffer);
    buffer += sizeof(T) * 256;

    // Group size buffer
    T h_group_size[CSRGEMM_MAXGROUPS];
    T h_group_offset[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(T) * CSRGEMM_MAXGROUPS);

    // Permutation array
    T* d_perm = nullptr;
    // T *total1;
    // total1 = (T *)malloc(sizeof(T)*(m+1)); 
    // If maximum of intermediate products exceeds 32, we process the rows in groups of
    // similar sized intermediate products
    if(int_max > 32)
    {
        // Group size buffer
        T* d_group_size = reinterpret_cast<T*>(buffer);
        buffer += sizeof(T) * 256 * CSRGEMM_MAXGROUPS;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        csrgemm_group_reduce_part1<CSRGEMM_DIM, CSRGEMM_MAXGROUPS><<<
                           dim3(CSRGEMM_DIM),
                           dim3(CSRGEMM_DIM)>>>
                           (m,
                           csr_row_ptr_C,
                           d_group_size);

        csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS><<<
                           dim3(1),
                           dim3(CSRGEMM_DIM)>>>
                           (d_group_size);
#undef CSRGEMM_DIM
        // Exclusive sum to obtain group offsets
        cub::DeviceScan::ExclusiveSum(nullptr, 
        cudaPrim_size, d_group_size, d_group_offset, CSRGEMM_MAXGROUPS);
        cudaPrim_buffer = reinterpret_cast<void*>(buffer);
        cub::DeviceScan::ExclusiveSum(cudaPrim_buffer, 
        cudaPrim_size, d_group_size, d_group_offset, CSRGEMM_MAXGROUPS);
        cudaDeviceSynchronize();
        // Copy group sizes to host
        cudaMemcpy(&h_group_size,
                    d_group_size,
                    sizeof(T) * CSRGEMM_MAXGROUPS,
                    cudaMemcpyDeviceToHost);
        // cudaMemcpy(&h_group_offset,
        //             d_group_offset,
        //             sizeof(T) * CSRGEMM_MAXGROUPS,
        //             cudaMemcpyDeviceToHost);
        // for(int i = 0; i < CSRGEMM_MAXGROUPS; i++)
        // {
        //     printf("i %d h_group_size %d h_group_offset %d\n", i, h_group_size[i], h_group_offset[i]);
        // }
        // Permutation temporary arrays
        T* tmp_vals = reinterpret_cast<T*>(buffer);
        buffer += ((sizeof(T) * m - 1) / 256 + 1) * 256;

        T* tmp_perm = reinterpret_cast<T*>(buffer);
        buffer += ((sizeof(T) * m - 1) / 256 + 1) * 256;

        T* tmp_keys = reinterpret_cast<T*>(buffer);
        buffer += ((sizeof(T) * m - 1) / 256 + 1) * 256;

#define IDENTITY_DIM 512
    dim3 identity_blocks((m - 1) / IDENTITY_DIM + 1);
    dim3 identity_threads(IDENTITY_DIM);

    identity_kernel<IDENTITY_DIM><<< identity_blocks, identity_threads>>>( m, tmp_perm);
#undef IDENTITY_DIM
        cudaDeviceSynchronize();
        
        // cudaMemcpy(total1, csr_row_ptr_C, (m) * sizeof(T), cudaMemcpyDeviceToHost);
        // cudaMemcpy(total, tmp_perm, (m) * sizeof(T), cudaMemcpyDeviceToHost);
        // printf("before\n");
        // for(int i = 0; i < m; i ++)
        // {
        //     printf("i %d tmp_perm %d csr_row_ptr_C %d\n",i, total[i], total1[i]);
        // }
        cub::DoubleBuffer<T> d_keys(csr_row_ptr_C, tmp_keys);
        cub::DoubleBuffer<T> d_vals(tmp_perm, tmp_vals);
        // Sort pairs (by groups)        
        cub::DeviceRadixSort::SortPairs(nullptr, cudaPrim_size, d_keys, d_vals, m, 0, 3);
        cudaPrim_buffer = reinterpret_cast<void*>(buffer);
        cub::DeviceRadixSort::SortPairs(cudaPrim_buffer, cudaPrim_size, d_keys, d_vals, m, 0, 3);
        cudaDeviceSynchronize();
        d_perm = d_vals.Current();
        // T * d_k = d_keys.Current();
        // cudaMemcpy(total1, d_k, (m) * sizeof(T), cudaMemcpyDeviceToHost);
        // cudaMemcpy(total, d_perm, (m) * sizeof(T), cudaMemcpyDeviceToHost);
        // printf("after\n");
        // for(int i = 0; i < m; i ++)
        // {
        //     printf("i %d d_perm %d csr_row_ptr_C %d\n",i, total[i], total1[i]);
        // }
        // Release tmp_keys buffer
        buffer -= ((sizeof(T) * m - 1) / 256 + 1) * 256;
        // cudaMemcpy(csr_row_ptr_C, total1, (m) * sizeof(T), cudaMemcpyHostToDevice);
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        cudaMemset(d_group_offset, 0, sizeof(T));
    }

    // Compute non-zero entries per row for each group

    // Group 0: 0 - 32 intermediate products
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 4
#define CSRGEMM_HASHSIZE 32
        
            csrgemm_nnz_wf_per_row<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_HASHSIZE, CSRGEMM_NNZ_HASH><<<
            dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM)>>>
            (h_group_size[0],
            &d_group_offset[0],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            true,
            false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();
    // T * csr_row_ptr_C1 = NULL;
    // cudaMalloc((void **)&csr_row_ptr_C1, (m + 1)*sizeof(T));
    // // Exclusive sum to obtain row pointers of C
    // cub::DeviceScan::ExclusiveSum(nullptr, 
    // cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C1, m + 1);
    // cudaPrim_buffer = reinterpret_cast<void*>(buffer);
    // cub::DeviceScan::ExclusiveSum(cudaPrim_buffer, 
    // cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C1, m + 1);
    // cudaMemcpy(&int_max, csr_row_ptr_C1 + m , sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h 0 int_max %d\n",int_max);
    // Group 1: 33 - 64 intermediate products
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 64
        
            csrgemm_nnz_wf_per_row<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_HASHSIZE, CSRGEMM_NNZ_HASH><<<
            dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM)>>>
            (h_group_size[1],
            &d_group_offset[1],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            true,
            false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();
    // cub::DeviceScan::ExclusiveSum(nullptr, 
    // cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C1, m + 1);
    // cudaPrim_buffer = reinterpret_cast<void*>(buffer);
    // cub::DeviceScan::ExclusiveSum(cudaPrim_buffer, 
    // cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C1, m + 1);
    // cudaMemcpy(&int_max, csr_row_ptr_C1 + m , sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h 1 int_max %d\n",int_max);
    // Group 2: 65 - 512 intermediate products
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 512
        csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                          CSRGEMM_SUB,
                          CSRGEMM_HASHSIZE,
                          CSRGEMM_NNZ_HASH><<<
                           dim3(h_group_size[2]),
                           dim3(CSRGEMM_DIM)>>>
                           (&d_group_offset[2],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           true,
                           false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();
    // cub::DeviceScan::ExclusiveSum(nullptr, 
    // cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C1, m + 1);
    // cudaPrim_buffer = reinterpret_cast<void*>(buffer);
    // cub::DeviceScan::ExclusiveSum(cudaPrim_buffer, 
    // cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C1, m + 1);
    // cudaMemcpy(&int_max, csr_row_ptr_C1 + m , sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h 2 int_max %d\n",int_max);
    // Group 3: 513 - 1024 intermediate products
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 1024
        csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                  CSRGEMM_SUB,
                                  CSRGEMM_HASHSIZE,
                                  CSRGEMM_NNZ_HASH><<<
                           dim3(h_group_size[3]),
                           dim3(CSRGEMM_DIM)>>>
                           (&d_group_offset[3],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           true,
                           false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    cudaDeviceSynchronize();
    // Group 4: 1025 - 2048 intermediate products
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 2048
        csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                  CSRGEMM_SUB,
                                  CSRGEMM_HASHSIZE,
                                  CSRGEMM_NNZ_HASH><<<
                           dim3(h_group_size[4]),
                           dim3(CSRGEMM_DIM)>>>
                          (&d_group_offset[4],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           true,
                           false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    cudaDeviceSynchronize();
    // Group 5: 2049 - 4096 intermediate products
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 4096
        csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                  CSRGEMM_SUB,
                                  CSRGEMM_HASHSIZE,
                                  CSRGEMM_NNZ_HASH><<<
                           dim3(h_group_size[5]),
                           dim3(CSRGEMM_DIM)>>>
                           (&d_group_offset[5],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           true,
                           false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 6: 4097 - 8192 intermediate products
    if(h_group_size[6] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 8192
        csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                  CSRGEMM_SUB,
                                  CSRGEMM_HASHSIZE,
                                  CSRGEMM_NNZ_HASH><<<
                           dim3(h_group_size[6]),
                           dim3(CSRGEMM_DIM)>>>
                           (&d_group_offset[6],
                           d_perm,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_row_ptr_C,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           ALPHA_SPARSE_INDEX_BASE_ZERO,
                           true,
                           false);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 7: more than 8192 intermediate products
    if(h_group_size[7] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_CHUNKSIZE 2048
        T* workspace_B = nullptr;

        // Allocate additional buffer for C = alpha * A * B
        cudaMalloc((void**)&workspace_B, sizeof(T) * nnz_A);

        csrgemm_nnz_block_per_row_multipass<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_CHUNKSIZE><<<
            dim3(h_group_size[7]),
            dim3(CSRGEMM_DIM)>>>
            (n,
            &d_group_offset[7],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            workspace_B,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            ALPHA_SPARSE_INDEX_BASE_ZERO,
            true,
            false);
        cudaDeviceSynchronize();
        cudaFree(workspace_B);

#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // Exclusive sum to obtain row pointers of C
    cub::DeviceScan::ExclusiveSum(nullptr, 
    cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C, m + 1);
    cudaPrim_buffer = reinterpret_cast<void*>(buffer);
    cub::DeviceScan::ExclusiveSum(cudaPrim_buffer, 
    cudaPrim_size, csr_row_ptr_C, csr_row_ptr_C, m + 1);
    cudaDeviceSynchronize();
    // Store nnz of C
    cudaMemcpy(nnz_C, csr_row_ptr_C + m, sizeof(T), cudaMemcpyDeviceToHost);
    // printf("nnz_Ced %d\n", *nnz_C);
    // Adjust nnz by index base
    *nnz_C -= 0;
    // cudaFree(temp_buffer);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
