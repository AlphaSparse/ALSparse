#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <iostream>

template <unsigned int BLOCKSIZE, typename T>
__device__ __forceinline__ void alphasparse_blockreduce_max(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = max(data[i], data[i + 512]); } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = max(data[i], data[i + 256]); } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = max(data[i], data[i + 128]); } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = max(data[i], data[i +  64]); } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = max(data[i], data[i +  32]); } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = max(data[i], data[i +  16]); } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = max(data[i], data[i +   8]); } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = max(data[i], data[i +   4]); } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = max(data[i], data[i +   2]); } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = max(data[i], data[i +   1]); } __syncthreads(); }
}

template <unsigned int BLOCKSIZE, typename T>
__device__ __forceinline__ void alphasparse_blockreduce_sum(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = data[i] + data[i + 512]; } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = data[i] + data[i + 256]; } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = data[i] + data[i + 128]; } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = data[i] + data[i +  64]; } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = data[i] + data[i +  32]; } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = data[i] + data[i +  16]; } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = data[i] + data[i +   8]; } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = data[i] + data[i +   4]; } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = data[i] + data[i +   2]; } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = data[i] + data[i +   1]; } __syncthreads(); }
}

template <int BLOCKSIZE, typename J>
__global__ static void csrcolor_reordering_identity(J size, J* identity)
{
    const J gid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    if(gid < size)
    {
        identity[gid] = gid;
    }
}

template <unsigned int BLOCKSIZE, typename J = int>
__global__ static void 
csrcolor_kernel_count_colors(J size,
                            const J* __restrict__ colors,
                            J* __restrict__ workspace)
{
    J gid = blockIdx.x * blockDim.x + threadIdx.x;
    J inc = gridDim.x * blockDim.x;

    __shared__ J sdata[BLOCKSIZE];

    J mx = 0;
    for(J idx = gid; idx < size; idx += inc)
    {
        J color = colors[idx];
        if(color > mx)
        {
            mx = color;
        }
    }

    sdata[threadIdx.x] = mx;

    __syncthreads();
    alphasparse_blockreduce_max<BLOCKSIZE>(threadIdx.x, sdata);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        workspace[blockIdx.x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE, typename J = int>
__global__ static void 
csrcolor_kernel_count_colors_finalize(J* __restrict__ workspace)
{
    __shared__ J sdata[BLOCKSIZE];

    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();
    alphasparse_blockreduce_max<BLOCKSIZE>(threadIdx.x, sdata);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        workspace[0] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE, typename J = int>
__global__ static void 
csrcolor_kernel_count_uncolored(J size,
                                const J* __restrict__ colors,
                                J* __restrict__ workspace)
{
    J gid = blockIdx.x * blockDim.x + threadIdx.x;
    J inc = gridDim.x * blockDim.x;

    __shared__ J sdata[BLOCKSIZE];

    J sum = 0;
    for(J idx = gid; idx < size; idx += inc)
    {
        if(colors[idx] == -1)
        {
            ++sum;
        }
    }

    sdata[threadIdx.x] = sum;

    __syncthreads();
    alphasparse_blockreduce_sum<BLOCKSIZE>(threadIdx.x, sdata);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        workspace[blockIdx.x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE, typename J = int>
__global__ static void 
csrcolor_kernel_count_uncolored_finalize(J* __restrict__ workspace)
{
    __shared__ J sdata[BLOCKSIZE];

    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();
    alphasparse_blockreduce_sum<BLOCKSIZE>(threadIdx.x, sdata);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        workspace[0] = sdata[0];
    }
}

static __forceinline__ __device__ uint32_t murmur3_32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

template <unsigned int BLOCKSIZE, typename I = int, typename J = int>
__global__ static void 
csrcolor_kernel_jpl(J m,
                    J color,
                    const I* __restrict__ csr_row_ptr,
                    const J* __restrict__ csr_col_ind,
                    alphasparseIndexBase_t csr_base,
                    J* __restrict__ colors)
{
    // Each thread processes a vertex
    J row = blockIdx.x * blockDim.x + threadIdx.x;
    // Do not run out of bounds
    if(row >= m)
    {
        return;
    }
    // Assume current vertex is maximum and minimum
    bool min = true, max = true;
    // Do not process already colored vertices
    if(colors[row] != -1)
    {        
        return;
    }
    // Get row weight
    uint32_t row_hash = murmur3_32(row);
    // Look at neighbors to check their random number
    const I bound = csr_row_ptr[row + 1] - csr_base;
    for(I j = csr_row_ptr[row] - csr_base; j < bound; ++j)
    {
        // Column to check against
        J col = csr_col_ind[j] - csr_base;
        // Skip diagonal
        if(row == col)
        {
            continue;
        }
        // Current neighbors color (-1 if uncolored)
        J color_nb = colors[col];

        // Skip already colored neighbors
        if(color_nb != -1 && color_nb != color && color_nb != (color + 1))
        {
            continue;
        }
        // Compute column hash
        uint32_t col_hash = murmur3_32(col);
        // Found neighboring vertex with larger weight,
        // vertex cannot be a maximum
        if(row_hash <= col_hash)
        {
            max = false;
        }
        // Found neighboring vertex with smaller weight,
        // vertex cannot be a minimum
        if(row_hash >= col_hash)
        {
            min = false;
        }
    }
    // If vertex is a maximum or a minimum then color it.
    if(max)
    {
        colors[row] = color;
    }
    else if(min)
    {
        colors[row] = color + 1;
    }
}

template <int NUMCOLUMNS_PER_BLOCK, int WF_SIZE, typename J>
__global__ static void 
csrcolor_assign_uncolored_kernel(J size, J m, J n, J shift_color, J* __restrict__ colors, J* __restrict__ index_sequence)
{
    static constexpr J  s_uncolored_value = static_cast<J>(-1);
    const int           wavefront_index   = threadIdx.x / WF_SIZE;
    const J             lane_index        = threadIdx.x % WF_SIZE;
    const uint64_t      filter            = 0xffffffffffffffff >> (63 - lane_index);
    const J             column_index      = NUMCOLUMNS_PER_BLOCK * blockIdx.x + wavefront_index;

    if(column_index < n)
    {
        J shift = shift_color + index_sequence[column_index];
        // The warp handles the entire column.
        for(J row_index = lane_index; row_index < m; row_index += WF_SIZE)
        {
            const J gid = column_index * m + row_index;
            // Get value.
            J* pcolor = colors + gid;
            // Predicate.
            const bool predicate = (gid < size) ? (s_uncolored_value == *pcolor) : false;
            // Mask of the wavefront.
            const uint64_t wavefront_mask = __ballot_sync(0xFFFFFFFF, predicate);
            // Get the number of previous non-zero in the row.
            const uint64_t count_previous_uncolored = __popcll(wavefront_mask & filter);
            // Synchronize for cache considerations.
            __syncthreads();

            if(predicate)
            {
                // Calculate local index.
                const uint64_t local_index = count_previous_uncolored - 1;
                // Populate the sparse matrix.
                *pcolor = shift + local_index;
            }
            // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
            // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
            shift += __shfl_sync(0xFFFFFFFF, count_previous_uncolored, WF_SIZE - 1, WF_SIZE);
        }
    }
}

template <int n, typename I>
static __forceinline__ __device__ void count_uncolored_reduce_device(int tx, I* sdata)
{
    __syncthreads();
    if(tx < n / 2)
    {
        sdata[tx] += sdata[tx + n / 2];
    }
    count_uncolored_reduce_device<n / 2>(tx, sdata);
}

template <>
__forceinline__ __device__ void count_uncolored_reduce_device<0, int32_t>(int tx,
                                                                          int32_t*      sdata)
{
}

template <int NB_X, typename J>
__global__ static void count_uncolored(
    J size, J m, J n, const J* __restrict__ colors, J* __restrict__ uncolored_per_sequence)
{
    static constexpr J s_uncolored_value = static_cast<J>(-1);

    J tx  = threadIdx.x;
    J col = blockIdx.x;

    J m_full = (m / NB_X) * NB_X;
    J res    = static_cast<J>(0);

    __shared__ J sdata[NB_X];

    colors += col * m + ((tx < m) ? tx : 0);
    for(J i = 0; i < m_full; i += NB_X)
    {
        res += (colors[i] == s_uncolored_value) ? 1 : 0;
    }

    if(tx + m_full < m)
    {
        res += (colors[m_full] == s_uncolored_value) ? 1 : 0;
    }

    sdata[tx] = res;
    if(NB_X > 16 && m >= NB_X)
    {
        count_uncolored_reduce_device<NB_X>(tx, sdata);
    }
    else
    {
        __syncthreads();

        if(tx == 0)
        {
            for(J i = 1; i < m && i < NB_X; i++)
                sdata[0] += sdata[i];
        }

        __syncthreads();
    }
    __syncthreads();
    if(tx == 0)
    {
        uncolored_per_sequence[col] = sdata[0];
    }
}