#include "alphasparse.h"
#include <iostream>
#include "alphasparse_csrcolor.h"
#include <cub/cub.cuh>

using namespace cub;

template <typename J>
static alphasparseStatus_t csrcolor_assign_uncolored(alphasparseHandle_t handle,
                                                        J                num_colored,
                                                        J                colors_length,
                                                        J*               colors)
{
    cudaStream_t stream = handle->stream;
    J                   m, n;

    J*                 seq_ptr = nullptr;
    static constexpr J NB      = 256;

    m = NB * 4;
    n = (colors_length - 1) / m + 1;

    cudaMalloc((void**)&seq_ptr, sizeof(J) * (n + 1));
    cudaMemsetAsync(seq_ptr, 0, sizeof(J) * (n + 1), stream);

    // Count uncolored values.
    {
        dim3 kernel_blocks(n);
        dim3 kernel_threads(NB);
        count_uncolored<NB, J><<<
                           kernel_blocks,
                           kernel_threads,
                           0,
                           stream>>>(
                           colors_length,
                           m,
                           n,
                           colors,
                           seq_ptr + 1);
    }
    // Next perform an inclusive sum.
    size_t temp_storage_bytes = 0;
    DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, seq_ptr, seq_ptr, n + 1, stream);

    bool  d_temp_alloc;
    void* d_temp_storage = nullptr;

    // Device buffer should be sufficient for rocprim in most cases
    // if(handle->buffer_size >= temp_storage_bytes)
    // {
    //     d_temp_storage = handle->buffer;
    //     d_temp_alloc   = false;
    // }
    // else
    {
        cudaMalloc((void **)&d_temp_storage, temp_storage_bytes);
        d_temp_alloc = true;
    }

    // Perform actual inclusive sum.
    DeviceScan::InclusiveSum(d_temp_storage,
                            temp_storage_bytes,
                            seq_ptr,
                            seq_ptr,
                            n + 1,
                            stream);

    // Free rocprim buffer, if allocated.
    if(d_temp_alloc == true)
    {
        cudaFree(d_temp_storage);
    }

    // Now we traverse again and we use num_colored_per_sequence.
    static constexpr int data_ratio = sizeof(J) / sizeof(float);
    if(1)
    {
        static constexpr int WF_SIZE            = 32;
        static constexpr int NCOLUMNS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        int                  blocks             = (n - 1) / NCOLUMNS_PER_BLOCK + 1;
        dim3                 k_blocks(blocks), k_threads(WF_SIZE * NCOLUMNS_PER_BLOCK);

        csrcolor_assign_uncolored_kernel<NCOLUMNS_PER_BLOCK, WF_SIZE><<<
                                            k_blocks,
                                            k_threads,
                                            0,
                                            stream>>>(
                                            colors_length,
                                            m,
                                            n,
                                            num_colored,
                                            colors,
                                            seq_ptr);
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename T, typename I, typename J, typename K>
alphasparseStatus_t
csrcolor_template(alphasparseHandle_t handle,
                J m,
                I nnz,
                const alphasparseMatDescr_t descrA,
                const void* csrValA,
                const I* csrRowPtrA,
                const J* csrColIndA,
                const void* fractionToColor,
                J* ncolors,
                J* coloring,
                J* reordering,
                alphasparseColorInfo_t info)
{
    if(descrA->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL) return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    
    static constexpr int blocksize = 256;

    cudaStream_t stream = handle->stream;
    *ncolors            = -2;

    J num_uncolored     = m;
    J max_num_uncolored = m - m * ((K *)fractionToColor)[0];

    // Create workspace.
    J* workspace;
    cudaMalloc((void**)&workspace, sizeof(J) * blocksize);
    // Initialize coloring
    cudaMemsetAsync(coloring, -1, sizeof(J) * m, stream);
    // Iterate until the desired fraction of colored vertices is reached
    while(num_uncolored > max_num_uncolored)
    {
        *ncolors += 2;
        // Run Jones-Plassmann Luby algorithm
        csrcolor_kernel_jpl<blocksize, I, J><<<
                           dim3((m - 1) / blocksize + 1),
                           dim3(blocksize),
                           0,
                           stream>>>(
                           m,
                           *ncolors,
                           csrRowPtrA,
                           csrColIndA,
                           descrA->base,
                           coloring);

        // Count colored vertices
        csrcolor_kernel_count_uncolored<blocksize, J><<<
                           dim3(blocksize),
                           dim3(blocksize),
                           0,
                           stream>>>(
                           m,
                           coloring,
                           workspace);

        // Gather results.
        csrcolor_kernel_count_uncolored_finalize<blocksize, J><<<
                           dim3(1),
                           dim3(blocksize),
                           0,
                           stream>>>
                          (workspace);
        cudaStreamSynchronize(stream);
        // Copy colored max vertices for current iteration to host
        cudaMemcpyAsync(&num_uncolored, workspace, sizeof(J), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // Need to count the number of coloring, compute the maximum value + 1.
    // This is something I'll need to figure out.
    // *ncolors += 2 is not the right number of coloring, sometimes yes, sometimes no.
    {
        csrcolor_kernel_count_colors<blocksize, J><<<
                           dim3(blocksize),
                           dim3(blocksize),
                           0,
                           stream>>>(
                           m,
                           coloring,
                           workspace);

        csrcolor_kernel_count_colors_finalize<blocksize, J><<<
                           dim3(1),
                           dim3(blocksize),
                           0,
                           stream>>>
                           (workspace);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(ncolors, workspace, sizeof(J), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        *ncolors += 1;
    }

    // Free workspace.
    cudaFree(workspace);

    if(num_uncolored > 0)
    {
        csrcolor_assign_uncolored(handle, *ncolors, m, coloring);
        *ncolors += num_uncolored;
    }

    // Calculating reorering if required.
    if(nullptr != reordering)
    {
        int* reordering_identity = nullptr;
        int* sorted_colors       = nullptr;

        // Create identity.
        cudaMalloc((void **)&reordering_identity, sizeof(J) * m);

        csrcolor_reordering_identity<1024, J><<<
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           stream>>>(
                           m,
                           reordering_identity);

        // Alloc output sorted coloring.
        cudaMalloc((void **)&sorted_colors, sizeof(J) * m);

        {
            int* keys_input    = coloring;
            int* values_input  = reordering_identity;
            int* keys_output   = sorted_colors;
            int* values_output = reordering;

            size_t temporary_storage_size_bytes;
            void*  temporary_storage_ptr = nullptr;

            // Get required size of the temporary storage
            DeviceRadixSort::SortPairs(temporary_storage_ptr,
                                      temporary_storage_size_bytes,
                                      keys_input,
                                      keys_output,
                                      values_input,
                                      values_output,
                                      m,
                                      0,
                                      sizeof(int) * 8,
                                      stream);

            // allocate temporary storage
            cudaMalloc((void **)&temporary_storage_ptr, temporary_storage_size_bytes);
            // perform sort
            DeviceRadixSort::SortPairs(temporary_storage_ptr,
                                      temporary_storage_size_bytes,
                                      keys_input,
                                      keys_output,
                                      values_input,
                                      values_output,
                                      m,
                                      0,
                                      sizeof(int) * 8,
                                      stream);

            cudaFree(temporary_storage_ptr);
        }

        cudaFree(reordering_identity);
        cudaFree(sorted_colors);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseScsrcolor(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    const float* csrValA,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    const float* fractionToColor,
                    int* ncolors,
                    int* coloring,
                    int* reordering,
                    alphasparseColorInfo_t info)
{
    csrcolor_template<float, int, int, float>(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseDcsrcolor(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    const double* csrValA,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    const double* fractionToColor,
                    int* ncolors,
                    int* coloring,
                    int* reordering,
                    alphasparseColorInfo_t info)
{
    csrcolor_template<double, int, int, double>(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseCcsrcolor(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    const void* csrValA,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    const float* fractionToColor,
                    int* ncolors,
                    int* coloring,
                    int* reordering,
                    alphasparseColorInfo_t info)
{
    csrcolor_template<cuFloatComplex, int, int, float>(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseZcsrcolor(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    const void* csrValA,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    const double* fractionToColor,
                    int* ncolors,
                    int* coloring,
                    int* reordering,
                    alphasparseColorInfo_t info)
{
    csrcolor_template<cuDoubleComplex, int, int, double>(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}