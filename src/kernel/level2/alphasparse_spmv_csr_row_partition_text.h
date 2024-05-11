#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_csr_vector.h"

template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row_f(
    const IndexType num_rows,
    const IndexType nnz,
    const IndexType ind,
    IndexType &row,
    IndexType &row_end,
    const IndexType row_predict,
    const IndexType row_predict_end,
    const IndexType *__restrict__ row_ptr)
{
    if (!overflow || ind < nnz)
    {
        if (ind >= row_end)
        {
            row = row_predict;
            row_end = row_predict_end;
            if (ind < row_end)
            {
                return;
            }
            row_end = row_ptr[++row + 1];
            if (ind < row_end)
            {
                return;
            }
            row_end = row_ptr[++row + 1];
            if (ind < row_end)
            {
                return;
            }
            row_end = row_ptr[++row + 1];
            if (ind < row_end)
            {
                return;
            }

            int right = 1;
            row_end = row_ptr[row + 2];
            while (ind >= row_end)
            {
                right *= 2;
                if (row + right >= num_rows)
                {
                    right = num_rows - row - 1;
                    break;
                }
                row_end = row_ptr[row + right + 1];
            }
            if (right == 1)
            {
                ++row;
                return;
            }
            // 二分查找
            int left = right / 2;
            while (left <= right)
            {
                int mid = (right + left) / 2;
                row_end = row_ptr[row + mid + 1];
                if (ind >= row_end)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
            row += left;
            row_end = row_ptr[row + 1];
            return;
            // while (ind >= row_end)
            // {
            // row_end = row_ptr[++row + 1];
            // }
        }
    }
    else
    {
        row = num_rows - 1;
        row_end = nnz;
    }
}

template <bool last, unsigned subwarp_size = 16,
          typename IndexType, typename Closure>
__device__ __forceinline__ void process_window_f(
    const cooperative_groups::thread_block_tile<subwarp_size> &group,
    const IndexType num_rows, const IndexType nnz, IndexType ind,
    IndexType &row, IndexType &row_end, IndexType &nrow, IndexType &nrow_end,
    float &temp_val, const float *val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ csr_row_ptr, const float *b,
    float *c, Closure scale, cudaTextureObject_t tex)
{
    const auto curr_row = row;
    const auto col = col_idxs[ind];

    find_next_row_f<last>(num_rows, nnz, ind, row, row_end, nrow, nrow_end,
                        csr_row_ptr);
    // asm("prefetch.global.L2 [%0];"::"l"(p_b));
    float val_ind = val[ind];
    float b_col   = tex1Dfetch<float>(tex, col);//b[col];
    // segmented scan
    if (group.any(curr_row != row))
    {
        warp_atomic_add(group, curr_row != row, temp_val, curr_row, c, scale);
        // if(curr_row != row) 
        // {
        //     atomicAdd(&(c[curr_row]), scale(temp_val));
        //     temp_val = 0.0f;
        // }
        nrow = group.shfl(row, subwarp_size - 1);
        nrow_end = group.shfl(row_end, subwarp_size - 1);
    }
    
    if (!last || ind < nnz)
    { 
        temp_val += val_ind * b_col;
    }
}

template <typename T, typename Closure>
__device__ __forceinline__ void load_balance_spmv_kernel_f(
    T nwarps, const T m, const T nnz,
    const float *val, const T *col_idxs,
    const T *csr_row_ptr, T *srow,
    const float *b, float *c, Closure scale,
    const T warps_in_block, const T warp_size, cudaTextureObject_t tex)
{
    const T warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    if (warp_idx >= nwarps)
    {
        return;
    }
    //not very heavy
    
    const T start = get_warp_start_idx(nwarps, nnz, warp_idx, warp_size);
    constexpr T wsize = 32;
    const T end =
        min(get_warp_start_idx(nwarps, nnz, warp_idx + 1, warp_size),
            ceildivT<T>(nnz, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = csr_row_ptr[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;

    float temp_val = 0.0f;
    T ind = start + threadIdx.x;
    //
    find_next_row_f<true>(m, nnz, ind, row, row_end, nrow, nrow_end,
                        csr_row_ptr);
    
    const T ind_end = end - wsize;
    const cooperative_groups::thread_block_tile<wsize> tile_block =
        tiled_partition<wsize>(this_thread_block());
    //0.41ms
    for (; ind < ind_end; ind += wsize)
    {   
        process_window_f<false>(tile_block, m, nnz, ind, row,
                              row_end, nrow, nrow_end, temp_val, val, col_idxs,
                              csr_row_ptr, b, c, scale, tex);
    } 
    //0.78
    process_window_f<true>(tile_block, m, nnz, ind, row, row_end,
                         nrow, nrow_end, temp_val, val, col_idxs, csr_row_ptr, b,
                         c, scale, tex);
    //1.28                     
    warp_atomic_add(tile_block, true, temp_val, row, c, scale);
    // atomicAdd(&c[row], scale(temp_val));
    //1.55ms
    return ;
}


template <typename T>
__global__ __launch_bounds__(spmv_block_size) void abstract_load_balance_spmv_f(
    T nwarps, const T m, const T nnz,
    const float *val, const T *col_idxs,
    const T *csr_row_ptr, T *srow,
    const float *b, float *c, const float alpha,
    const T warps_in_block, const T warp_size, cudaTextureObject_t tex)
{
    load_balance_spmv_kernel_f(
        nwarps, m, nnz, val, col_idxs, csr_row_ptr, srow, b, c,
        [&alpha](const float &x)
        {
            return static_cast<float>(alpha * x);
        },
        warps_in_block, warp_size, tex);
}

template <typename T>
static void load_balance_spmv_f(const T m,
                              const T n,
                              const T nnz,
                              const float alpha,
                              T *srow,
                              const T *csr_row_ptr,
                              const T *csr_col_ind,
                              const float *csr_val,
                              const float *x,
                              const float beta,
                              float *y,
                              T nwarps,
                              const T warp_size,
                              const int warps_in_block,
                              cudaTextureObject_t tex)
{
    if (nwarps > 0)
    {
        const dim3 csr_block(warp_size, warps_in_block, 1);
        const dim3 csr_grid(ceildivT((int64_t)nwarps, (int64_t)warps_in_block), 1);
        if (csr_grid.x > 0 && csr_grid.y > 0)
        {
            T *srow_device = srow;
            // cudaMalloc((void **)&srow_device, nwarps * sizeof(T));
            // cudaMemcpy(srow_device,
            //            srow,
            //            nwarps * sizeof(T),
            //            cudaMemcpyHostToDevice);
            abstract_load_balance_spmv_f<<<csr_grid, csr_block>>>(
                nwarps, m, nnz, csr_val,
                csr_col_ind, csr_row_ptr, srow_device, x, y,
                alpha, warps_in_block, warp_size, tex);
            // printf("\n+++++++++++++++++++++%d,%d,%d,%d,%d,%d,%d\n", nwarps, warps_in_block, warp_size, csr_grid.x, csr_grid.y, csr_block.x, csr_block.y);
            // printf("\n+++++++++++++++++++++%d,%d,%d,%d\n", csr_grid.x, csr_grid.y, csr_block.x, csr_block.y);
        }
    }
}

template <typename T, T warp_size>
__device__ static T lower_bound_int2_f(T l, T r, int64_t target, T nwarps, cudaTextureObject_t tex)
{
    while (l <= r)
    {
        int m = (l + r) / 2;
        auto index = tex1Dfetch<T>(tex, m+1);

        if (ceildivT<T>(index, warp_size) * (int64_t)nwarps < target)
        {
            l = m + 1;
        }
        else
        {
            r = m - 1;
        }
    }

    return l;
}

template <typename T, typename W, typename V, T warp_size>
__global__ static void balanced_partition_row_by_nnz_and_scale_y_f(
    const T *acc_sum_arr,
    T rows,
    T nwarps,
    T *partition,
    int64_t ave,
    const T ave_row,
    const W beta,
    V *y,
    cudaTextureObject_t tex)
{
    extern __shared__ int64_t shared_buffer[];
    const T gid = blockIdx.x * blockDim.x + threadIdx.x;
    T index = min(threadIdx.x * ave_row, rows - 1) + 1;
    // auto row_ind = tex1Dfetch<T>(tex, index);

    // shared_buffer[threadIdx.x] = ceildivT<T>(row_ind, warp_size) * (int64_t)nwarps;
    shared_buffer[threadIdx.x] = ceildivT<T>(acc_sum_arr[index - 1], warp_size) * (int64_t)nwarps;
    __syncthreads();
    if (gid >= nwarps)
    {
        if (gid < nwarps + rows)
        {
            y[gid - nwarps] *= beta;
        }
        return;
    }
    T idx2 = lower_bound_int6<T>(shared_buffer, 0, blockDim.x, (ave * gid));
    T left = max(min(idx2 * ave_row, rows), 0);
    T right = min(left + ave_row, rows);
    // partition[gid] = lower_bound_int2_f<T, warp_size>(left, right, (ave * gid), nwarps, tex);
    partition[gid] = lower_bound_int2<T, warp_size>(acc_sum_arr, left, right, (ave * gid), nwarps);
}

template <typename T>
alphasparseStatus_t spmv_csr_load_f(alphasparseHandle_t handle,
                                  T m,
                                  T n,
                                  T nnz,
                                  const float alpha,
                                  const float *csr_val,
                                  T *csr_row_ptr,
                                  const T *csr_col_ind,
                                  float *x,
                                  const float beta,
                                  float *y,
                                  void *externalBuffer)
{
    const T SM = 520;
    const T MAX_WARP_PER_SM = 64;
    const T warp_size = 32;
    const T nwarps_ = SM * MAX_WARP_PER_SM / warp_size;
    const T warps_in_block = 8;
    T nwarps = clac_size(nnz, warp_size, nwarps_);

    const T BLOCK_SIZE = 32 * warps_in_block;

    T *partition = (T *)externalBuffer;
    const int maxbytes = BLOCK_SIZE * sizeof(int64_t);
    const int64_t ave = ceildivT<T>(nnz, warp_size);
    const T ave_row = ceildivT<T>(m, BLOCK_SIZE);

    cudaResourceDesc resDesc_row;
    memset(&resDesc_row, 0, sizeof(resDesc_row));
    resDesc_row.resType = cudaResourceTypeLinear;
    resDesc_row.res.linear.devPtr = csr_row_ptr;
    resDesc_row.res.linear.desc.f = cudaChannelFormatKindSigned;
    resDesc_row.res.linear.desc.x = 32; // bits per channel
    resDesc_row.res.linear.sizeInBytes = m*sizeof(T);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex_row=0;
    cudaCreateTextureObject(&tex_row, &resDesc_row, &texDesc, NULL);

    {
        const T GRIDSIZE = ceildivT<T>(nwarps + m, BLOCK_SIZE);
        balanced_partition_row_by_nnz_and_scale_y_f<T, float, float, warp_size><<<dim3(GRIDSIZE), dim3(BLOCK_SIZE), maxbytes, handle->stream>>>(
            csr_row_ptr + 1, m, nwarps, partition, ave, ave_row, beta, y, tex_row);
    }
    // create texture object
    cudaResourceDesc resDescx;
    memset(&resDescx, 0, sizeof(resDescx));
    resDescx.resType = cudaResourceTypeLinear;
    resDescx.res.linear.devPtr = x;
    resDescx.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDescx.res.linear.desc.x = 32; // bits per channel
    resDescx.res.linear.sizeInBytes = m*sizeof(float);

    // cudaResourceDesc resDescy;
    // memset(&resDescy, 0, sizeof(resDesc));
    // resDescy.resType = cudaResourceTypeLinear;
    // resDescy.res.linear.devPtr = y;
    // resDescy.res.linear.desc.f = cudaChannelFormatKindFloat;
    // resDescy.res.linear.desc.x = 32; // bits per channel
    // resDescy.res.linear.sizeInBytes = m*sizeof(float);    

    // create texture object: we only have to do this once!
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex=0;
    cudaCreateTextureObject(&tex, &resDescx, &texDesc, NULL);
    // cudaTextureObject_t tey=0;
    // cudaCreateTextureObject(&tey, &resDescy, &texDesc, NULL);
    load_balance_spmv_f(m, n, nnz, alpha, partition, csr_row_ptr,
                    csr_col_ind, csr_val, x, beta, y,
                    nwarps, warp_size, warps_in_block, tex);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
