#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"


#define WARP_SIZE   32
#define BLOCKSIZE 128

// Implementation motivated by papers 'Efficient Sparse Matrix-Vector Multiplication on CUDA',
// 'Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors' and
// 'Segmented operations for sparse matrix computation on vector multiprocessors'

template <typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCKSIZE) static __global__ void coomvn_general_wf_reduce(T nnz,
                                                                             T loops,
                                                                             W alpha,
                                                                             const T *__restrict__ coo_row_ind,
                                                                             const T *__restrict__ coo_col_ind,
                                                                             const U *__restrict__ coo_val,
                                                                             const U *__restrict__ x,
                                                                             V *__restrict__ y,
                                                                             T *__restrict__ row_block_red,
                                                                             V *__restrict__ val_block_red)
{
    T tid = threadIdx.x;
    T gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    // Lane index (0,...,WARP_SIZE)
    T lid = tid & (WARP_SIZE - 1); // lid = tid % WARP_SIZE
    // Warp index
    T wid = gid / WARP_SIZE;

    // Initialize block buffers
    if (lid == 0) {
        *(row_block_red + wid) = -1;
        *(val_block_red + wid) = V{};
    }

    // Global COO array index start for current wavefront
    T offset = wid * loops * WARP_SIZE;

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ T shared_row[BLOCKSIZE];
    __shared__ V shared_val[BLOCKSIZE];

    // Initialize shared memory
    shared_row[tid] = -1;
    *(val_block_red + wid) = V{};

    __syncthreads();

    // Quick return when thread is out of bounds
    if (offset + lid >= nnz) {
        return;
    }

    T row;
    V val;

    // Current threads index into COO structure
    T idx = offset + lid;

    // Each thread processes 'loop' COO entries
    while (idx < offset + loops * WARP_SIZE) {
        // Get corresponding COO entry, if not out of bounds.
        // This can happen when processing more than 1 entry if
        // nnz % WARP_SIZE != 0
        if (idx < nnz) {
            row            = *(coo_row_ind + idx);
            V v = *(coo_val + idx);
            val = alpha * v;
            V tmp = *(x + *(coo_col_ind + idx));
            val = val * tmp;
        } else {
            row = -1;
            val = V{};
        }

        // First thread in wavefront checks row index from previous loop
        // if it has been completed or if additional rows have to be
        // appended.
        if (idx > offset && lid == 0) {
            T prevrow = shared_row[tid + WARP_SIZE - 1];
            if (row == prevrow) {
                val = val + shared_val[tid + WARP_SIZE - 1];
            } else if (prevrow >= 0) {
                y[prevrow] = y[prevrow] + shared_val[tid + WARP_SIZE - 1];
            }
        }

        __syncthreads();

        // Update shared buffers
        shared_row[tid] = row;
        shared_val[tid] = val;

        __syncthreads();

        // #pragma unroll
        // Segmented wavefront reduction
        for (T j = 1; j < WARP_SIZE; j <<= 1) {
            if (lid >= j) {
                if (row == shared_row[tid - j]) {
                    val = val + shared_val[tid - j];
                }
            }
            __syncthreads();

            shared_val[tid] = val;

            __syncthreads();
        }

        // All lanes but the last one write their result in y.
        // The last value might need to be appended by the next iteration.
        if (lid < WARP_SIZE - 1) {
            if (row != shared_row[tid + 1] && row >= 0) {
                y[row] = y[row] + val;
            }
        }

        // Keep going for the next iteration
        idx += WARP_SIZE;
    }

    // Write last entries into buffers for segmented block reduction
    if (lid == WARP_SIZE - 1) {
        *(row_block_red + wid) = row;
        *(val_block_red + wid) = val;
    }
}

template <typename T, typename U, typename V>
// Segmented block reduction kernel
static __device__ void segmented_blockreduce(const T *__restrict__ rows, V *__restrict__ vals)
{
    T tid = threadIdx.x;

    // #pragma unroll
    for (T j = 1; j < BLOCKSIZE; j <<= 1) {
        V val = {};
        if (tid >= j) {
            if (rows[tid] == rows[tid - j]) {
                val = vals[tid - j];
            }
        }
        __syncthreads();

        vals[tid] = vals[tid] + val;
        __syncthreads();
    }
}

// Do the final block reduction of the block reduction buffers back into global memory
template <typename T, typename U, typename V>
__launch_bounds__(BLOCKSIZE) __global__ static void coomvn_general_block_reduce(
    T nnz,
    const T *__restrict__ row_block_red,
    const V *__restrict__ val_block_red,
    V *__restrict__ y)
{
    T tid = threadIdx.x;

    // Quick return when thread is out of bounds
    if (tid >= nnz) {
        return;
    }

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ T shared_row[BLOCKSIZE];
    __shared__ V shared_val[BLOCKSIZE];

    // Loop over blocks that are subject for segmented reduction
    for (T i = tid; i < nnz; i += BLOCKSIZE) {
        // Copy data to reduction buffers
        shared_row[tid] = row_block_red[i];
        shared_val[tid] = val_block_red[i];

        __syncthreads();

        // Do segmented block reduction
        segmented_blockreduce<T, U, V>(shared_row, shared_val);

        // Add reduced sum to y if valid
        T row   = shared_row[tid];
        T rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

        if (row != rowp1 && row >= 0) {
            y[row] = y[row] + shared_val[tid];
        }

        __syncthreads();
    }
}

template <typename T, typename W, typename V>
__launch_bounds__(1024) __global__ static void mulbeta(T m,
                                                       const W beta,
                                                       V *__restrict__ y)
{
    T tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;
    y[tid] = y[tid] * beta;
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmv_coo(alphasparseHandle_t handle,
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
    const T threadPerBlock = 1024;
    const T blockPerGrid   = m / threadPerBlock + 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mulbeta<T, W, V>), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, beta, y);

#define COOMVN_DIM 128
    T maxthreads = handle->properties.maxThreadsPerBlock;
    T nprocs     = handle->properties.multiProcessorCount;
    T maxblocks  = (nprocs * maxthreads - 1) / COOMVN_DIM + 1;

    const T wavefront_size = 32;

    T minblocks = (nnz - 1) / COOMVN_DIM + 1;
    T nblocks   = maxblocks < minblocks ? maxblocks : minblocks;
    T nwfs      = nblocks * (COOMVN_DIM / wavefront_size);
    T nloops    = (nnz / wavefront_size + 1) / nwfs + 1;

    dim3 coomvn_blocks(nblocks);
    dim3 coomvn_threads(COOMVN_DIM);

    // Buffer
    char *ptr = (char *)(handle->buffer);
    ptr += 256;

    // row block reduction buffer
    T *row_block_red = (T *)(ptr);
    ptr += ((sizeof(T) * nwfs - 1) / 256 + 1) * 256;

    // val block reduction buffer
    V *val_block_red = (V *)ptr;

    // wavefront_size == 64)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(coomvn_general_wf_reduce<T, U, V, W>), coomvn_blocks, coomvn_threads, 0, handle->stream, nnz,
                       nloops,
                       alpha,
                       coo_row_ind,
                       coo_col_ind,
                       coo_val,
                       x,
                       y,
                       row_block_red,
                       val_block_red);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(coomvn_general_block_reduce<T, U, V>), dim3(1), coomvn_threads, 0, handle->stream, nwfs,
                       row_block_red,
                       val_block_red,
                       y);
#undef COOMVN_DIM

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

