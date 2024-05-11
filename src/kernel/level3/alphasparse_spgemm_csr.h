#include "alphasparse.h"
#include <iostream>
#include "csrspgemm_device.h"
#include <cub/cub.cuh>  

template <typename I,
          typename J,
          typename T,
          typename std::enable_if<
              std::is_same<T, double2>::value
                  || (std::is_same<T, double>::value && std::is_same<I, int64_t>::value
                      && std::is_same<J, int64_t>::value)
                  || (std::is_same<T, float2>::value
                      && std::is_same<I, int64_t>::value && std::is_same<J, int64_t>::value)
                  || (std::is_same<T, float2>::value
                    && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value))
                  || (std::is_same<T, double>::value
                    && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)),
              int>::type
          = 0>
static inline alphasparseStatus_t csrgemm_launcher(alphasparseHandle_t     handle,
                                                J                    group_size,
                                                const J*             group_offset,
                                                const J*             perm,
                                                J                    m,
                                                J                    n,
                                                J                    k,
                                                const T              alpha,
                                                const I*             csr_row_ptr_A,
                                                const J*             csr_col_ind_A,
                                                const T*             csr_val_A,
                                                const I*             csr_row_ptr_B,
                                                const J*             csr_col_ind_B,
                                                const T*             csr_val_B,
                                                const T              beta,
                                                const I*             csr_row_ptr_D,
                                                const J*             csr_col_ind_D,
                                                const T*             csr_val_D,
                                                const I*             csr_row_ptr_C,
                                                J*                   csr_col_ind_C,
                                                T*                   csr_val_C,
                                                alphasparseIndexBase_t base_A,
                                                alphasparseIndexBase_t base_B,
                                                alphasparseIndexBase_t base_C,
                                                alphasparseIndexBase_t base_D,
                                                bool                 mul,
                                                bool                 add)
{
    return ALPHA_SPARSE_STATUS_INTERNAL_ERROR;
}

// template <typename I,
//           typename J,
//           typename T,
//           typename std::enable_if<
//               std::is_same<T, float2>::value
//                 && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)
//               || (std::is_same<T, double>::value
//                 && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)),
//               int>::type
//           = 0>
// static inline alphasparseStatus_t csrgemm_launcher(alphasparseHandle_t     handle,
//                                                     J                    group_size,
//                                                     const J*             group_offset,
//                                                     const J*             perm,
//                                                     J                    m,
//                                                     J                    n,
//                                                     J                    k,
//                                                     const T              alpha,
//                                                     const I*             csr_row_ptr_A,
//                                                     const J*             csr_col_ind_A,
//                                                     const T*             csr_val_A,
//                                                     const I*             csr_row_ptr_B,
//                                                     const J*             csr_col_ind_B,
//                                                     const T*             csr_val_B,
//                                                     const T              beta,
//                                                     const I*             csr_row_ptr_D,
//                                                     const J*             csr_col_ind_D,
//                                                     const T*             csr_val_D,
//                                                     const I*             csr_row_ptr_C,
//                                                     J*                   csr_col_ind_C,
//                                                     T*                   csr_val_C,
//                                                     alphasparseIndexBase_t base_A,
//                                                     alphasparseIndexBase_t base_B,
//                                                     alphasparseIndexBase_t base_C,
//                                                     alphasparseIndexBase_t base_D,
//                                                     bool                 mul,
//                                                     bool                 add)
// {
//     return ALPHA_SPARSE_STATUS_INTERNAL_ERROR;
// }

template <typename I,
          typename J,
          typename T,
          typename std::enable_if<
              std::is_same<T, float>::value && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value),
                  // || (std::is_same<T, double>::value
                  //     && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)),
                  // || (std::is_same<T, float2>::value
                  //     && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)),
              int>::type
          = 0>
static inline alphasparseStatus_t csrgemm_launcher(alphasparseHandle_t     handle,
                                                    J                    group_size,
                                                    const J*             group_offset,
                                                    const J*             perm,
                                                    J                    m,
                                                    J                    n,
                                                    J                    k,
                                                    const T              alpha,
                                                    const I*             csr_row_ptr_A,
                                                    const J*             csr_col_ind_A,
                                                    const T*             csr_val_A,
                                                    const I*             csr_row_ptr_B,
                                                    const J*             csr_col_ind_B,
                                                    const T*             csr_val_B,
                                                    const T              beta,
                                                    const I*             csr_row_ptr_D,
                                                    const J*             csr_col_ind_D,
                                                    const T*             csr_val_D,
                                                    const I*             csr_row_ptr_C,
                                                    J*                   csr_col_ind_C,
                                                    T*                   csr_val_C,
                                                    alphasparseIndexBase_t base_A,
                                                    alphasparseIndexBase_t base_B,
                                                    alphasparseIndexBase_t base_C,
                                                    alphasparseIndexBase_t base_D,
                                                    bool                 mul,
                                                    bool                 add)
{
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 64
#define CSRGEMM_HASHSIZE 4096
    csrgemm_fill_block_per_row_device<CSRGEMM_DIM,
                                      CSRGEMM_SUB,
                                      CSRGEMM_HASHSIZE,
                                      CSRGEMM_FLL_HASH><<<
                        dim3(group_size),
                        dim3(CSRGEMM_DIM),
                        0,
                        handle->stream>>>
                        (std::max(k, n),
                        group_offset,
                        perm,
                        alpha,
                        csr_row_ptr_A,
                        csr_col_ind_A,
                        csr_val_A,
                        csr_row_ptr_B,
                        csr_col_ind_B,
                        csr_val_B,
                        beta,
                        csr_row_ptr_D,
                        csr_col_ind_D,
                        csr_val_D,
                        csr_row_ptr_C,
                        csr_col_ind_C,
                        csr_val_C,
                        base_A,
                        base_B,
                        base_C,
                        base_D,
                        mul,
                        add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U>
alphasparseStatus_t
spgemm_csr(alphasparseHandle_t handle,
           T m,
           T n,
           T k, 
           const U alpha,
           const T* csr_row_ptr_A,
           const T* csr_col_ind_A,
           const U* csr_val_A,
           const T  nnz_A,
           const T* csr_row_ptr_B,
           const T* csr_col_ind_B,
           const U* csr_val_B,
           const U beta,
           const T* csr_row_ptr_D,
           T* csr_col_ind_D,
           U* csr_val_D,
           const T* csr_row_ptr_C,
           T* csr_col_ind_C,
           U* csr_val_C,
           alphasparseIndexBase_t base_A,
           alphasparseIndexBase_t base_B,
           alphasparseIndexBase_t base_C,
           alphasparseIndexBase_t base_D,
           bool mul,
           bool add,
           void * temp_buffer)
{
    constexpr bool exceeding_smem
        = std::is_same<U, float2>::value
          || (std::is_same<U, double>::value && std::is_same<T, int64_t>::value
              && std::is_same<T, int64_t>::value)
          || (std::is_same<U, double2>::value && std::is_same<T, int64_t>::value
              && std::is_same<T, int64_t>::value)
          || (std::is_same<U, double>::value && std::is_same<T, int32_t>::value
              && std::is_same<T, int32_t>::value);

    cudaStream_t stream = handle->stream;
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // cudaPrim buffer
    size_t cudaPrim_size;
    void*  cudaPrim_buffer;

    // Determine maximum non-zero entries per row of all rows
    T* workspace = reinterpret_cast<T*>(buffer);

#define CSRGEMM_DIM 256
    csrgemm_max_row_nnz_part1<CSRGEMM_DIM><<<
                       dim3(CSRGEMM_DIM),
                       dim3(CSRGEMM_DIM)>>>
                       (m,
                       csr_row_ptr_C,
                       workspace);

    csrgemm_max_row_nnz_part2<CSRGEMM_DIM><<<dim3(1), dim3(CSRGEMM_DIM)>>>(workspace);
#undef CSRGEMM_DIM

    T nnz_max;
    
    cudaMemcpyAsync(&nnz_max, workspace, sizeof(T), cudaMemcpyDeviceToHost, stream);
    
    // Wait for host transfer to finish
    cudaStreamSynchronize(stream);
    // printf("nnz_max %d\n",nnz_max);
    // Group offset buffer
    T* d_group_offset = reinterpret_cast<T*>(buffer);
    buffer += sizeof(T) * 256;

    // Group size buffer
    T h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(T) * CSRGEMM_MAXGROUPS);

    // Permutation array
    T* d_perm = nullptr;

    // If maximum of row nnz exceeds 16, we process the rows in groups of
    // similar sized row nnz
    if(nnz_max > 16)
    {
        // Group size buffer
        T* d_group_size = reinterpret_cast<T*>(buffer);
        buffer += sizeof(T) * 256 * CSRGEMM_MAXGROUPS;

        // Permutation temporary arrays
        T* tmp_vals = reinterpret_cast<T*>(buffer);
        buffer += ((sizeof(T) * m - 1) / 256 + 1) * 256;

        T* tmp_perm = reinterpret_cast<T*>(buffer);
        buffer += ((sizeof(T) * m - 1) / 256 + 1) * 256;

        int* tmp_keys = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        int* tmp_groups = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        
        csrgemm_group_reduce_part2<CSRGEMM_DIM, CSRGEMM_MAXGROUPS, exceeding_smem><<<
        dim3(CSRGEMM_DIM),
        dim3(CSRGEMM_DIM)>>>
        (m,
        csr_row_ptr_C,
        d_group_size,
        tmp_groups);

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
        cudaMemcpyAsync(&h_group_size,
                          d_group_size,
                          sizeof(T) * CSRGEMM_MAXGROUPS,
                          cudaMemcpyDeviceToHost,
                          stream);

        // Wait for host transfer to finish
        cudaStreamSynchronize(stream);
        // for(int i = 0; i < CSRGEMM_MAXGROUPS; i++)
        //     std::cout<<"h_group_size  "<<i<<" "<<h_group_size[i]<<std::endl;
        // Create identity permutation for group access
        
#define IDENTITY_DIM 512
        dim3 identity_blocks((m - 1) / IDENTITY_DIM + 1);
        dim3 identity_threads(IDENTITY_DIM);

        identity_kernel<IDENTITY_DIM><<< identity_blocks, identity_threads>>>( m, tmp_perm);
#undef IDENTITY_DIM
        // Sort pairs (by groups)
        cub::DoubleBuffer<T> d_keys(tmp_groups, tmp_keys);
        cub::DoubleBuffer<T> d_vals(tmp_perm, tmp_vals);
        // Sort pairs (by groups)        
        cub::DeviceRadixSort::SortPairs(nullptr, cudaPrim_size, d_keys, d_vals, m, 0, 3);
        cudaPrim_buffer = reinterpret_cast<void*>(buffer);
        cub::DeviceRadixSort::SortPairs(cudaPrim_buffer, cudaPrim_size, d_keys, d_vals, m, 0, 3);
        cudaDeviceSynchronize();
        d_perm = d_vals.Current();

        // Release tmp_groups buffer
        // buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        // buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        cudaMemsetAsync(d_group_offset, 0, sizeof(T), stream);
    }

    // Compute columns and accumulate values for each group

    // Group 0: 0 - 16 non-zeros per row
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
        csrgemm_fill_wf_per_row_device<CSRGEMM_DIM,
                                      CSRGEMM_SUB,
                                      CSRGEMM_HASHSIZE,
                                      CSRGEMM_FLL_HASH><<<
                                      dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                                      dim3(CSRGEMM_DIM)>>>
                                      (h_group_size[0],
                                      std::max(k, n),
                                      &d_group_offset[0],
                                      d_perm,
                                      alpha,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      csr_val_A,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      csr_val_B,
                                      beta,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      csr_val_D,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      csr_val_C,
                                      base_A,
                                      base_B,
                                      base_C,
                                      base_D,
                                      mul,
                                      add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();printf("379\n");
    // Group 1: 17 - 32 non-zeros per row
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32
        csrgemm_fill_wf_per_row_device<CSRGEMM_DIM,
                                        CSRGEMM_SUB,
                                        CSRGEMM_HASHSIZE,
                                        CSRGEMM_FLL_HASH><<<
                                        dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                                        dim3(CSRGEMM_DIM),
                                        0,
                                        stream>>>
                                        (h_group_size[1],
                                        std::max(k, n),
                                        &d_group_offset[1],
                                        d_perm,
                                        alpha,
                                        csr_row_ptr_A,
                                        csr_col_ind_A,
                                        csr_val_A,
                                        csr_row_ptr_B,
                                        csr_col_ind_B,
                                        csr_val_B,
                                        beta,
                                        csr_row_ptr_D,
                                        csr_col_ind_D,
                                        csr_val_D,
                                        csr_row_ptr_C,
                                        csr_col_ind_C,
                                        csr_val_C,
                                        base_A,
                                        base_B,
                                        base_C,
                                        base_D,
                                        mul,
                                        add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();printf("422 h_group_size[2] %d\n", h_group_size[2]);
    // Group 2: 33 - 256 non-zeros per row
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
        csrgemm_fill_block_per_row_device<CSRGEMM_DIM,
                                          CSRGEMM_SUB,
                                          CSRGEMM_HASHSIZE,
                                          CSRGEMM_FLL_HASH><<<
                                          dim3(h_group_size[2]),
                                          dim3(CSRGEMM_DIM),
                                          0,
                                          stream>>>
                                          (std::max(k, n),
                                          &d_group_offset[2],
                                          d_perm,
                                          alpha,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          csr_val_A,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          csr_val_B,
                                          beta,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          csr_val_D,
                                          csr_row_ptr_C,
                                          csr_col_ind_C,
                                          csr_val_C,
                                          base_A,
                                          base_B,
                                          base_C,
                                          base_D,
                                          mul,
                                          add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();printf("464\n");
    // Group 3: 257 - 512 non-zeros per row
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
        csrgemm_fill_block_per_row_device<CSRGEMM_DIM,
                                          CSRGEMM_SUB,
                                          CSRGEMM_HASHSIZE,
                                          CSRGEMM_FLL_HASH><<<
                                          dim3(h_group_size[3]),
                                          dim3(CSRGEMM_DIM),
                                          0,
                                          stream>>>
                                          (std::max(k, n),
                                          &d_group_offset[3],
                                          d_perm,
                                          alpha,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          csr_val_A,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          csr_val_B,
                                          beta,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          csr_val_D,
                                          csr_row_ptr_C,
                                          csr_col_ind_C,
                                          csr_val_C,
                                          base_A,
                                          base_B,
                                          base_C,
                                          base_D,
                                          mul,
                                          add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();printf("506\n");
    // Group 4: 513 - 1024 non-zeros per row
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
        csrgemm_fill_block_per_row_device<CSRGEMM_DIM,
                                          CSRGEMM_SUB,
                                          CSRGEMM_HASHSIZE,
                                          CSRGEMM_FLL_HASH><<<
                                          dim3(h_group_size[4]),
                                          dim3(CSRGEMM_DIM),
                                          0,
                                          stream>>>
                                          (std::max(k, n),
                                          &d_group_offset[4],
                                          d_perm,
                                          alpha,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          csr_val_A,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          csr_val_B,
                                          beta,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          csr_val_D,
                                          csr_row_ptr_C,
                                          csr_col_ind_C,
                                          csr_val_C,
                                          base_A,
                                          base_B,
                                          base_C,
                                          base_D,
                                          mul,
                                          add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();printf("548\n");
    // Group 5: 1025 - 2048 non-zeros per row
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
        csrgemm_fill_block_per_row_device<CSRGEMM_DIM,
                                          CSRGEMM_SUB,
                                          CSRGEMM_HASHSIZE,
                                          CSRGEMM_FLL_HASH><<<
                                          dim3(h_group_size[5]),
                                          dim3(CSRGEMM_DIM),
                                          0,
                                          stream>>>
                                          (std::max(k, n),
                                          &d_group_offset[5],
                                          d_perm,
                                          alpha,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          csr_val_A,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          csr_val_B,
                                          beta,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          csr_val_D,
                                          csr_row_ptr_C,
                                          csr_col_ind_C,
                                          csr_val_C,
                                          base_A,
                                          base_B,
                                          base_C,
                                          base_D,
                                          mul,
                                          add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();printf("590\n");
#ifndef rocsparse_ILP64
    // Group 6: 2049 - 4096 non-zeros per row
    if(h_group_size[6] > 0 && !exceeding_smem)
    {
        csrgemm_launcher(handle,
                          h_group_size[6],
                          &d_group_offset[6],
                          d_perm,
                          m,
                          n,
                          k,
                          alpha,
                          csr_row_ptr_A,
                          csr_col_ind_A,
                          csr_val_A,
                          csr_row_ptr_B,
                          csr_col_ind_B,
                          csr_val_B,
                          beta,
                          csr_row_ptr_D,
                          csr_col_ind_D,
                          csr_val_D,
                          csr_row_ptr_C,
                          csr_col_ind_C,
                          csr_val_C,
                          base_A,
                          base_B,
                          base_C,
                          base_D,
                          mul,
                          add);
    }
#endif
    // cudaDeviceSynchronize();
    // printf("625\n");
    // Group 7: more than 4096 non-zeros per row
    if(h_group_size[7] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_CHUNKSIZE 2048
        T* workspace_B = nullptr;
        // T nnz_A ;
        // cudaMemcpy(&nnz_A, csr_row_ptr_A + m, sizeof(T), cudaMemcpyDeviceToHost);
        // printf("nnz_A %d\n", nnz_A);
        if(mul == true)
        {
            // Allocate additional buffer for C = alpha * A * B
            cudaMalloc((void**)&workspace_B, sizeof(T) * nnz_A);
        }

        csrgemm_fill_block_per_row_multipass_device<CSRGEMM_DIM,
                                                    CSRGEMM_SUB,
                                                    CSRGEMM_CHUNKSIZE><<<
                                                    dim3(h_group_size[7]),
                                                    dim3(CSRGEMM_DIM),
                                                    0,
                                                    stream>>>
                                                    (n,
                                                    &d_group_offset[7],
                                                    d_perm,
                                                    alpha,
                                                    csr_row_ptr_A,
                                                    csr_col_ind_A,
                                                    csr_val_A,
                                                    csr_row_ptr_B,
                                                    csr_col_ind_B,
                                                    csr_val_B,
                                                    beta,
                                                    csr_row_ptr_D,
                                                    csr_col_ind_D,
                                                    csr_val_D,
                                                    csr_row_ptr_C,
                                                    csr_col_ind_C,
                                                    csr_val_C,
                                                    workspace_B,
                                                    base_A,
                                                    base_B,
                                                    base_C,
                                                    base_D,
                                                    mul,
                                                    add);

        if(mul == true)
        {
            cudaFree(workspace_B);
        }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }
    // cudaDeviceSynchronize();
    // printf("685\n");
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
