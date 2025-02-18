#pragma once
#include <iostream>

#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"
#include "alphasparse/util_dcu.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include "alphasparse/kernel_dcu.h"
#ifdef __cplusplus
}
#endif /*__cplusplus */

/**
 * dataset: U.Fl. sparse matrices
 * average speedup(AS) compared with csr-vector.
 * but large ITEMS_PER_THREAD can reduce preprocess time(AVT).
 * 
 * ITEMS_PER_THREAD=1, AS: 1.8549 
 * ITEMS_PER_THREAD=3, AS: 1.7334, AVT: 0.37ms  
 * ITEMS_PER_THREAD=4, AS: 1.6872  
 * ITEMS_PER_THREAD=5, AS: 1.5531  
 * ITEMS_PER_THREAD=6, AS: 1.5431  
 * ITEMS_PER_THREAD=7, AS: 1.4811, AVT: 0.22ms
 * ITEMS_PER_THREAD=8, AS: 1.4088  
 * ITEMS_PER_THREAD=9, AS: 1.3422  
 */

#define BLOCK_THREADS    256
#define ITEMS_PER_THREAD 3

struct CoordinateT {
    ALPHA_INT x;
    ALPHA_INT y;
};

static alphasparse_status_t csr_merge_coocase_dispatch(alphasparseHandle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       ALPHA_INT nnz,
                                                       const ALPHA_Number alpha,
                                                       const ALPHA_Number *csr_val,
                                                       const ALPHA_INT *csr_row_ptr,
                                                       const ALPHA_INT *csr_col_ind,
                                                       const ALPHA_Number *x,
                                                       const ALPHA_Number beta,
                                                       ALPHA_Number *y,
                                                       u_int32_t flag,
                                                       alphasparse_dcu_mat_info_t info);

template <ALPHA_INT BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void get_row_indx_device(ALPHA_INT m,
                             const ALPHA_INT *csr_row_ptr,
                             ALPHA_INT *row_indx)
{
    ALPHA_INT tid = hipBlockIdx_x * BLOCKSIZE + tid;

    if (tid >= m) return;

    for (uint32_t j = csr_row_ptr[tid]; j < csr_row_ptr[tid + 1]; j++) {
        row_indx[j] = tid;
    }
}

__launch_bounds__(1024) __global__ static void mulbeta(ALPHA_INT m,
                                                       const ALPHA_Number beta,
                                                       ALPHA_Number *__restrict__ y)
{
    ALPHA_INT tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid >= m) return;

    alpha_mul(y[tid], y[tid], beta);
}

// Segmented block reduction kernel
static __device__ void segmented_blockreduce(const ALPHA_INT *__restrict__ rows, ALPHA_Number *__restrict__ vals)
{
    ALPHA_INT tid = hipThreadIdx_x;

    // #pragma unroll
    for (ALPHA_INT j = 1; j < BLOCK_THREADS; j <<= 1) {
        ALPHA_Number val;
        alpha_setzero(val);
        if (tid >= j) {
            if (rows[tid] == rows[tid - j]) {
                val = vals[tid - j];
            }
        }
        __syncthreads();

        //vals[tid] = vals[tid] + val;
        alpha_add(vals[tid], vals[tid], val);
        __syncthreads();
    }
}

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
__host__ __device__ __forceinline__ static void MergePathSearch(
    ALPHA_INT diagonal,
    const ALPHA_INT *a,
    // const ALPHA_INT *b,
    ALPHA_INT a_len,
    ALPHA_INT b_len,
    CoordinateT &path_coordinate)
{
    ALPHA_INT split_min = max(diagonal - b_len, 0);
    ALPHA_INT split_max = min(diagonal, a_len);

    while (split_min < split_max) {
        ALPHA_INT split_pivot = (split_min + split_max) >> 1;
        // if (a[split_pivot] <= b[diagonal - split_pivot - 1]) {
        if (a[split_pivot] <= diagonal - split_pivot - 1) {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        } else {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    path_coordinate.x = min(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}

/**
 * Spmv search kernel. Identifies merge path starting coordinates for each tile.
 */
__global__ static void DeviceSpmvSearchKernel(
    int num_merge_tiles, ///< [in] Number of SpMV merge tiles (spmv grid size)
    CoordinateT *d_tile_coordinates, ///< [out] Pointer to the temporary array of tile starting coordinates
    ALPHA_INT num_rows,
    ALPHA_INT num_nonzeros,
    const ALPHA_INT *d_row_end_offsets) ///< [in] SpMV input parameter bundle
{
    /// Constants
    const ALPHA_INT TILE_ITEMS = ITEMS_PER_THREAD;

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    int tile_idx = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    if (tile_idx < num_merge_tiles + 1) {
        ALPHA_INT diagonal = (tile_idx * TILE_ITEMS);
        CoordinateT tile_coordinate;

        // Search the merge path
        MergePathSearch(diagonal, d_row_end_offsets, num_rows, num_nonzeros, tile_coordinate);

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;
    }

    if (tile_idx == num_merge_tiles + 1) {
        d_tile_coordinates[num_merge_tiles].x = num_rows;
        d_tile_coordinates[num_merge_tiles].y = num_nonzeros;
    }
}

/**
 * Spmv kernel.
 */
__global__ static void DeviceSpmvKernel(ALPHA_INT m,
                                        ALPHA_Number alpha,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        const ALPHA_Number *csr_val,
                                        const ALPHA_Number *x,
                                        ALPHA_Number beta,
                                        ALPHA_Number *y,
                                        ALPHA_INT num_merge_tiles,
                                        const CoordinateT *path_coordinate,
                                        ALPHA_Number *reduc_val,
                                        ALPHA_INT *reduc_row)
{
    const ALPHA_INT bid = hipBlockIdx_x;
    const ALPHA_INT gid = (hipBlockIdx_x * BLOCK_THREADS) + threadIdx.x;
    const ALPHA_INT tid = threadIdx.x;

    __shared__ ALPHA_INT s_row[BLOCK_THREADS];
    __shared__ ALPHA_Number s_val[BLOCK_THREADS];

    if (gid >= num_merge_tiles) return;

    ALPHA_INT row_idx_start = path_coordinate[gid].x;
    ALPHA_INT row_idx_end   = path_coordinate[gid + 1].x;

    ALPHA_INT ai           = path_coordinate[gid].y;
    const ALPHA_INT ai_end = path_coordinate[gid + 1].y;

    s_val[tid] = ALPHA_ZERO;
    s_row[tid] = -1;
    __syncthreads();

    ALPHA_Number sum;
    ALPHA_INT row = row_idx_start;
    for (; row < row_idx_end; row++) {
        sum = ALPHA_ZERO;
        for (; ai < csr_row_ptr[row + 1]; ai++) {
            alpha_madde(sum, csr_val[ai], x[csr_col_ind[ai]]);
        }
        //! alpha_mule(y[row], beta);
        alpha_madde(y[row], sum, alpha);
    }

    sum = ALPHA_ZERO;
    for (; ai < ai_end; ai++) {
        alpha_madde(sum, csr_val[ai], x[csr_col_ind[ai]]);
    }
    alpha_mule(sum, alpha);
    alpha_adde(s_val[tid], sum);
    s_row[tid] = row;
    __syncthreads();

    //! inner block reduction 1
    // Segmented block reduction
    for (ALPHA_INT j = 1; j < BLOCK_THREADS; j <<= 1) {
        if (tid >= j) {
            if (row == s_row[tid - j]) {
                // sum = sum + s_val[tid - j];
                alpha_add(sum, sum, s_val[tid - j]);
            }
        }
        __syncthreads();

        s_val[tid] = sum;

        __syncthreads();
    }
    // All thread but the last one write their result in y.
    if (tid < BLOCK_THREADS - 1) {
        if (row != s_row[tid + 1] && row >= 0) {
            // y[row] = y[row] + sum;
            alpha_adde(y[row], sum);
        }
    }

    //! inner block reduction 2
    // segmented_blockreduce(s_row, s_val);
    // // Add reduced sum to y if valid
    // ALPHA_INT rowp1 = (tid < BLOCK_THREADS - 1) ? s_row[tid + 1] : -1;

    // if (tid < BLOCK_THREADS-1 && row != rowp1 && row >= 0 && row < m) {
    //     //y[row] = y[row] + shared_val[tid];
    //     alpha_add(y[row], y[row], s_val[tid]);
    // }

    //! inner block reduction 3
    // if (tid < BLOCK_THREADS - 1) {
    //     alpha_atomic_add(y[s_row[tid]], s_val[tid]);
    //     reduc_val[bid] = ALPHA_ZERO;
    //     reduc_row[bid] = -1;
    // }

    // for inter block reduction
    if (tid == BLOCK_THREADS - 1) {
        //! alpha_mule(y[row], beta);
        reduc_val[bid] = sum;
        reduc_row[bid] = row;
    }
}

// Do the final block reduction of the block reduction buffers back into global memory
__launch_bounds__(BLOCK_THREADS) __global__ static void csrmv_merge_general_block_reduce(
    ALPHA_INT nnz,
    ALPHA_INT m,
    const ALPHA_INT *__restrict__ row_block_red,
    const ALPHA_Number *__restrict__ val_block_red,
    ALPHA_Number *__restrict__ y)
{
    ALPHA_INT tid = hipThreadIdx_x;

    // Quick return when thread is out of bounds
    if (tid >= nnz) {
        return;
    }

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ ALPHA_INT shared_row[BLOCK_THREADS];
    __shared__ ALPHA_Number shared_val[BLOCK_THREADS];

    // Loop over blocks that are subject for segmented reduction
    for (ALPHA_INT i = tid; i < nnz; i += BLOCK_THREADS) {
        // Copy data to reduction buffers
        shared_row[tid] = row_block_red[i];
        shared_val[tid] = val_block_red[i];

        __syncthreads();

        // Do segmented block reduction
        segmented_blockreduce(shared_row, shared_val);

        // Add reduced sum to y if valid
        ALPHA_INT row   = shared_row[tid];
        ALPHA_INT rowp1 = (tid < BLOCK_THREADS - 1) ? shared_row[tid + 1] : -1;

        if (row != rowp1 && row >= 0 && row < m) {
            //y[row] = y[row] + shared_val[tid];
            alpha_add(y[row], y[row], shared_val[tid]);
        }

        __syncthreads();
    }
}

/**
 * merge-based csrmv: merge based load balance.
 * Merrill D, Garland M. 
 * Merge-based sparse matrix-vector multiplication (spmv) using the csr storage format
 * ACM SIGPLAN Notices, 2016, 51(8): 1-2.
 * 
 * worktiems: (m + 1) + nnz, row_ptr + values.
 * csr2coo can achieve similar results.
 * 
 * if 1 item per thread, same as coo, merge path can convert rowptr to rowindx fast
 * 
 * balance memory access: coo & csr
 * sizeof(row_ptr) + sizeof(val) + len(col_ind) vs len(row_ind) + len(val) + len(col_ind)
 * 
 * balance calculate: coo & csr
 * calculate item for each wavefront. 
 * segment reduce & block reduce for coo vs wavefront reduce for csr
 * 
 */
alphasparse_status_t csr_gemv_merge_dispatch(alphasparseHandle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const ALPHA_Number alpha,
                                             const ALPHA_Number *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const ALPHA_Number *x,
                                             const ALPHA_Number beta,
                                             ALPHA_Number *y,
                                             u_int32_t flag,
                                             alphasparse_dcu_mat_info_t info)
{
    // dispatch coo case
    if (ITEMS_PER_THREAD == 1) {
        return csr_merge_coocase_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag, info);
    }

    if (n == 1) {
        return csr_gemv_scalar_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    }

    const ALPHA_INT work_item       = m + nnz;
    const ALPHA_INT num_merge_tiles = CEIL(work_item, ITEMS_PER_THREAD);
    const ALPHA_INT num_blocks      = CEIL(num_merge_tiles, BLOCK_THREADS);

    if (num_blocks <= 1) {
        return csr_gemv_vector_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    }

    CoordinateT *d_tile_coordinates = nullptr;
    ALPHA_INT *reduc_row            = nullptr;
    ALPHA_Number *reduc_val         = nullptr;

    // printf("m:%d n:%d num_merge_tiles:%d, ITEMS_PER_THREAD:%d\n", m, nnz, num_merge_tiles, ITEMS_PER_THREAD);

    if (!info || !info->csrmv_info) {
        printf("infor or csrmv_infor nullptr.\n");
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    alphasparse_dcu_csrmv_info_t csrmv_info = info->csrmv_info;

    if (!csrmv_info->csr_merge_has_tuned) {
        double time = get_time_us();

        hipMalloc(&d_tile_coordinates, sizeof(CoordinateT) * (num_merge_tiles + 1));

        hipLaunchKernelGGL((DeviceSpmvSearchKernel),
                           dim3(CEIL(num_merge_tiles, 512)),
                           dim3(512),
                           0,
                           handle->stream,
                           num_merge_tiles,
                           d_tile_coordinates,
                           m,
                           nnz,
                           csr_row_ptr + 1);

        hipMalloc(&reduc_row, sizeof(ALPHA_INT) * num_blocks);
        hipMalloc(&reduc_val, sizeof(ALPHA_Number) * num_blocks);

        csrmv_info->csr_merge_has_tuned = true;
        csrmv_info->coordinate          = (void *)d_tile_coordinates;
        csrmv_info->num_merge_tiles     = num_merge_tiles;
        csrmv_info->reduc_val           = (void *)reduc_val;
        csrmv_info->reduc_row           = (void *)reduc_row;

        // time = (get_time_us() - time) / (1e3);
        // printf("preprocess:%f\n", time);
    }

    if (!csrmv_info->coordinate) {
        printf("merge nullptr.");
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    d_tile_coordinates = (CoordinateT *)csrmv_info->coordinate;
    reduc_val          = (ALPHA_Number *)csrmv_info->reduc_val;
    reduc_row          = (ALPHA_INT *)csrmv_info->reduc_row;

    hipLaunchKernelGGL(mulbeta, dim3(CEIL(m, 1024)), dim3(1024), 0, handle->stream, m, beta, y);

    hipLaunchKernelGGL((DeviceSpmvKernel),
                       dim3(num_blocks),
                       dim3(BLOCK_THREADS),
                       0,
                       handle->stream,
                       m,
                       alpha,
                       csr_row_ptr,
                       csr_col_ind,
                       csr_val,
                       x,
                       beta,
                       y,
                       num_merge_tiles,
                       d_tile_coordinates,
                       reduc_val,
                       reduc_row);

    hipLaunchKernelGGL(csrmv_merge_general_block_reduce,
                       dim3(1),
                       dim3(BLOCK_THREADS),
                       0,
                       handle->stream,
                       num_blocks,
                       m,
                       reduc_row,
                       reduc_val,
                       y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

/**
 *  coo case for ITEMS_PER_THREAD == 1
 * 
 */
static alphasparse_status_t csr_merge_coocase_dispatch(alphasparseHandle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       ALPHA_INT nnz,
                                                       const ALPHA_Number alpha,
                                                       const ALPHA_Number *csr_val,
                                                       const ALPHA_INT *csr_row_ptr,
                                                       const ALPHA_INT *csr_col_ind,
                                                       const ALPHA_Number *x,
                                                       const ALPHA_Number beta,
                                                       ALPHA_Number *y,
                                                       u_int32_t flag,
                                                       alphasparse_dcu_mat_info_t info)
{
    if (!info || !info->csrmv_info) {
        printf("infor or csrmv_infor nullptr.\n");
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    alphasparse_dcu_csrmv_info_t csrmv_info = info->csrmv_info;

    if (!csrmv_info->csr_merge_has_tuned) {
        ALPHA_INT *d_row_indx;
        hipMalloc(&d_row_indx, sizeof(ALPHA_INT) * nnz);

        const ALPHA_INT BLOCKSIZE = 512;
        hipLaunchKernelGGL((get_row_indx_device<BLOCKSIZE>),
                           dim3(CEIL(m, BLOCKSIZE)),
                           dim3(BLOCKSIZE),
                           0,
                           handle->stream,
                           m,
                           csr_row_ptr,
                           d_row_indx);

        csrmv_info->csr_merge_has_tuned = true;
        csrmv_info->coordinate          = (void *)d_row_indx;
    }

    if (!csrmv_info->coordinate) {
        printf("merge nullptr.");
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    dcu_gemv_coo(handle,
                 m,
                 n,
                 nnz,
                 alpha,
                 csr_val,
                 (ALPHA_INT *)csrmv_info->coordinate,
                 csr_col_ind,
                 x,
                 beta,
                 y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#undef BLOCK_THREADS
#undef ITEMS_PER_THREAD
