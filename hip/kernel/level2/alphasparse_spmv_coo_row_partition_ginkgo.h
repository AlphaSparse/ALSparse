#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_csr_vector.h"
#include <thrust/scan.h>

constexpr int coo_spmv_block_size = 128;

template <typename T>
T calculate_nwarps(const T nnz, const T warp_size)
{
    size_t warps_per_sm = 2;
    size_t nwarps_in_cuda = 80 * warps_per_sm;
    size_t multiple = 8;
    if (nnz >= 2e8)
    {
        multiple = 2048;
    }
    else if (nnz >= 2e7)
    {
        multiple = 512;
    }
    else if (nnz >= 2e6)
    {
        multiple = 128;
    }
    else if (nnz >= 2e5)
    {
        multiple = 32;
    }
    return std::min(multiple * nwarps_in_cuda,
                    size_t(ceildivT<T>(nnz, warp_size)));
}

template <int subwarp_size = 32, typename ValueType,
          typename IndexType, typename ValueType2, typename Closure>
__device__ void coo_spmv_ginkgo_kernel(const IndexType warp_size,
                                       const IndexType nnz,
                                       const IndexType num_lines,
                                       const ValueType *__restrict__ val,
                                       const IndexType *__restrict__ col,
                                       const IndexType *__restrict__ row,
                                       const ValueType *__restrict__ b,
                                       ValueType2 *__restrict__ c,
                                       Closure scale)
{
    ValueType temp_val = {};
    const auto start = blockDim.x * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    const auto column_id = blockIdx.y;
    IndexType num = (nnz > start) * ceildivT<IndexType>(nnz - start, subwarp_size);
    if (num_lines < num)
        num = num_lines;
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * subwarp_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block = tiled_partition<subwarp_size>(this_thread_block());
    for (; ind < ind_end; ind += subwarp_size)
    {
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] + column_id]
                                : ValueType{};
        auto next_row =
            (ind + subwarp_size < nnz) ? row[ind + subwarp_size] : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row))
        {
            bool is_first_in_segment = segment_scan<subwarp_size>(
                tile_block, curr_row, temp_val);
            if (is_first_in_segment)
            {
                atomicAdd(&(c[curr_row + column_id]),
                          scale(temp_val));
            }
            temp_val = ValueType{};
        }
        curr_row = next_row;
    }
    if (num > 0)
    {
        ind = ind_end;
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] + column_id]
                                : ValueType{};
        // segmented scan
        bool is_first_in_segment = segment_scan<subwarp_size>(
            tile_block, curr_row, temp_val);
        if (is_first_in_segment)
        {
            atomicAdd(&(c[curr_row + column_id]), scale(temp_val));
        }
    }
}

template <typename T, typename U, typename V, typename W>
__global__ __launch_bounds__(coo_spmv_block_size) void abstract_coo_spmv_ginkgo(
    const T warp_size, T nnz, T num_lines, const W alpha,
    const U *val, const T *col,
    const T *row,
    const U *b, V *c)
{
    coo_spmv_ginkgo_kernel(
        warp_size, nnz, num_lines, val, col, row, b, c,
        [&alpha](const V &x)
        { return static_cast<V>(alpha * x); });
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmv_coo_ginkgo(alphasparseHandle_t handle,
                T m,
                T n,
                T nnz,
                const W alpha,
                const U *coo_val,
                const T *coo_row_ind,
                const T *coo_col_ind,
                const U *x,
                const W beta,
                V *y)
{
    const T SM = 80;
    const T MAX_WARP_PER_SM = 64;
    const T warp_size = 32;
    const T warps_in_block = 4;
    T nwarps = calculate_nwarps(nnz, warp_size);
    printf("nwarps:%d\n", nwarps);

    const dim3 coo_block(warp_size, warps_in_block, 1);
    const dim3 coo_grid(ceildivT<T>(nwarps, warps_in_block), 1);
    int num_lines = ceildivT<T>(nnz, nwarps * warp_size);
    if (beta != W{})
    {
        int block_size = 512;
        int grid_size = ceildivT<int>(m, block_size);
        hipLaunchKernelGGL(array_scale, grid_size, block_size, 0, 0, m, y, beta);
    }
    else
    {
        hipMemset(y, 0, sizeof(V) * m);
    }
    hipLaunchKernelGGL(abstract_coo_spmv_ginkgo, coo_grid, coo_block, 0, 0, 
        warp_size, nnz, num_lines, alpha, coo_val,
        coo_col_ind,
        coo_row_ind,
        x,
        y);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
