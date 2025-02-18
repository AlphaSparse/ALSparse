#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <iostream>

#define FULL_MASK 0xffffffff
#define RefThreadPerBlock 256
#define PHYSICAL_WARP_SIZE 32

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

template<typename T, unsigned int warp_size>
__device__ __forceinline__ T warpReduce(T sum) {
  if (warp_size < 32) sum += __shfl_down_sync(FULL_MASK, sum, 16);
  if (warp_size < 16) sum += __shfl_down_sync(FULL_MASK, sum, 8);
  if (warp_size < 8) sum += __shfl_down_sync(FULL_MASK, sum, 4);
  if (warp_size < 4) sum += __shfl_down_sync(FULL_MASK, sum, 2);
  if (warp_size < 2) sum += __shfl_down_sync(FULL_MASK, sum, 1);
  return sum;
}

template<typename T>
static __device__ __forceinline__
T binary_search_segment_number(const T *segoffsets, const T n_seg,
                                 const T elem_id) {
  T lo = 1, hi = n_seg, mid;
  while (lo <= hi) {
    mid = (lo + hi) >> 1;
    if (segoffsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return hi;
}

template <typename T>
static __forceinline__ __device__ void merge_path_search(
    const T diagonal, const T x_len, const T y_len,
    const T *__restrict__ a, const T offset_y,
    T *__restrict__ x, T *__restrict__ y)
{
    T x_min = max(diagonal - y_len, (T)0);
    T x_max = min(diagonal, x_len);
    T pivot;
    while (x_min < x_max)
    {
        pivot = (x_max + x_min) / 2;
        if (a[pivot] < offset_y + diagonal - pivot)
        {
            x_min = pivot + 1;
        }
        else
        {
            x_max = pivot;
        }
    }

    *x = min(x_min, x_len);
    *y = diagonal - *x;
}

template<int block_size,
         typename T>
__global__ void merge_path_search_kernel(
  T *start_xs,
  T *start_ys,
  T num_rows,
  T nnz,
  const T num_merge_items,
  const T items_per_thread,
  const T thread_num,
  const T *csr_row_ptr)
{
  __shared__ T merge_seg_ptr[block_size + 1];
  const T local_tid = threadIdx.x;
  const T tid = blockIdx.x * blockDim.x + threadIdx.x;

  const T stride = CEIL(num_rows, block_size);
  const T seg_row_idx = MIN(local_tid * stride, num_rows); 

  merge_seg_ptr[local_tid] = __ldg(&csr_row_ptr[seg_row_idx]) + seg_row_idx;
  if (local_tid == block_size - 1) merge_seg_ptr[block_size] = nnz + num_rows;

  __syncthreads();

  if (tid <= thread_num) {
    const T diagonal = MIN(items_per_thread * tid, num_merge_items);
    const T seg_start_id = binary_search_segment_number<T>(merge_seg_ptr, block_size, diagonal);

    const T seg_start_x = MIN(seg_start_id * stride, num_rows);
    const T seg_start_y = merge_seg_ptr[seg_start_id] - seg_start_x;
    const T seg_end_id = MIN(seg_start_id + 1, block_size);
    const T seg_end_x = MIN(seg_start_x + stride, num_rows);
    const T seg_end_y = merge_seg_ptr[seg_end_id] - seg_end_x;
    const T seg_diagonal = seg_start_x + seg_start_y;
    T local_start_x, local_start_y;
    const T local_diagonal = diagonal - seg_diagonal;
    merge_path_search(local_diagonal, seg_end_x - seg_start_x, seg_end_y - seg_start_y, csr_row_ptr + seg_start_x + 1,
                      seg_start_y, &local_start_x, &local_start_y);

    start_xs[tid] = seg_start_x + local_start_x;
    start_ys[tid] = seg_start_y + local_start_y;
  }
}

// assigns a thread to an output element.
template<int block_size,
         int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_rowbalance_rowmajor_seqreduce_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];

  T rowid = blockIdx.x * blockDim.y + threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;
  T local_offset = threadIdx.y * blockDim.x;
  T local_tid = local_offset + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  U col_val[factor] = {};
  W sum[factor] = {};
  if (rowid < M) {
    T rowstart = csr_row_ptr[rowid];
    T rowend = csr_row_ptr[rowid + 1];
    for (T i = rowstart; i < rowend; i += warp_size) {
      if (i + warp_laneid < rowend) {
        s_val[local_tid] = csr_val[i + warp_laneid];
        s_col_ind[local_tid] = csr_col_ind[i + warp_laneid];
      }


      for (T j = 0; j < warp_size; j++) {
        if (i + j >= rowend) break;
        T rowid_B = s_col_ind[local_offset + j];
        U val = s_val[local_offset + j];

        if (factor > 1) {
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                col_val[k] = matB[rowid_B * ldb + colid];
              } else {
                col_val[k] = matB[colid * ldb + rowid_B];
              }
            }
          }
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              sum[k] += val * col_val[k];
            }
          }
        } else {
          T colid = block_colid + warp_laneid;
          if (colid < N) {
            if (ORDER_ROW) {
              sum[0] += val * matB[rowid_B * ldb + colid];
            } else {
              sum[0] += val * matB[colid * ldb + rowid_B];
            }
          }
        }
      }
    }
    #pragma unroll
    for (int k = 0; k < factor; k++) {
      T colid = block_colid + k * warp_size + warp_laneid; 
      if (colid < N) {
        if (ORDER_ROW) {
          matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
        } else {
          matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
        }
      }
    }
  }
}

template<int block_size,
         typename T>
static __global__ void
elebalance_partition_kernel(
  T M, T nnz, T items_per_seg,
  const T* __restrict__ csr_row_ptr,
  T* seg_row_id)
{
  __shared__ T csr_seg_row_ptr[block_size + 1];

  T local_tid = threadIdx.x;
  T blockid = blockIdx.x;
  T tid = blockid * blockDim.x + local_tid;
  T stride = CEIL(M, block_size);

  T block_eleid_start = blockid * block_size * items_per_seg;
  T block_eleid_end = MIN((blockid + 1)* block_size * items_per_seg, nnz);


  T csr_row_ptr_idx = MIN(((local_tid + 1) * stride), M);
  if(local_tid == 0) {
    csr_seg_row_ptr[0] = 0;
  }
  csr_seg_row_ptr[local_tid + 1] = __ldg(&csr_row_ptr[csr_row_ptr_idx]);

  __syncthreads();

  T seg_eleid = tid * items_per_seg;

  if (seg_eleid < nnz) {
    T csr_seg_row_ptr_id = binary_search_segment_number<T>(csr_seg_row_ptr, block_size, seg_eleid);

    T low = csr_seg_row_ptr_id * stride;
    T hi = MIN(low + stride, M);
    seg_row_id[tid] = low + binary_search_segment_number<T>(csr_row_ptr + low, hi - low, seg_eleid);
    if (seg_eleid + items_per_seg >= nnz) {
      seg_row_id[tid + 1] = M;
    }
  }
}

template<int block_size,
         int warp_size,
         int work_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_elebalance_rowmajor_seqreduce_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_row_id)
{
  __shared__ U s_val[block_size / warp_size * work_size];
  __shared__ T s_col_ind[block_size / warp_size * work_size];
  __shared__ T s_seg_row_id[block_size / warp_size + 1];

  T block_segid = blockIdx.x * blockDim.y;
  T segid = block_segid + threadIdx.y;
  T element_per_seg = work_size;
  T eleid = segid * element_per_seg;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T local_segid = local_warpid;
  T block_eleid = blockIdx.x * blockDim.y * element_per_seg;
  T warp_laneid = local_tid & (warp_size - 1);
  U col_val[factor] = {};
  W sum[factor] = {};
  W zero = {};


  if ((local_tid < block_size / warp_size + 1) && (block_eleid + local_tid * work_size < nnz + work_size)) {
    s_seg_row_id[local_tid] = seg_row_id[block_segid + local_tid];
  }

  __syncthreads();

  if (warp_laneid < work_size && eleid + warp_laneid < nnz) {
    s_val[local_warpid * work_size + warp_laneid] = csr_val[eleid + warp_laneid];
    s_col_ind[local_warpid * work_size + warp_laneid] = csr_col_ind[eleid + warp_laneid];
  }

  if (eleid < nnz) {
    T seg_row_start = s_seg_row_id[local_segid];
    T seg_row_length = s_seg_row_id[local_segid + 1] - seg_row_start;
    T rowid = seg_row_start;
    T step = csr_row_ptr[rowid + 1] - eleid; //remaining elements in this row.
    for (T i = 0; i < element_per_seg; i++) {
      if (eleid >= nnz) break;
      T s_eleid = eleid - block_eleid;

      T rowid_B = s_col_ind[s_eleid];
      if (factor > 1) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              col_val[k] = matB[rowid_B * ldb + colid];
            } else {
              col_val[k] = matB[colid * ldb + rowid_B];
            }
          }
        }

        if (i < step) {
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              sum[k] += s_val[s_eleid] * col_val[k];
            }
          }
          eleid++;
        } else {
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
              } else {
                atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
              }
            }
          }
          // next element may cross multirows.
          T seg_rowid = binary_search_segment_number<T>(csr_row_ptr + seg_row_start, seg_row_length, eleid);
          rowid = seg_row_start + seg_rowid;
          step += csr_row_ptr[rowid + 1] - eleid;
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              sum[k] = zero;
              sum[k] += s_val[s_eleid] * col_val[k];
            }
          }
          eleid++;
        }
      } else {
        T colid = block_colid + warp_laneid;
        if (colid < N) {
          if (i < step) {
            if (ORDER_ROW) {
              sum[0] += s_val[s_eleid] * matB[rowid_B * ldb + colid];
            } else {
              sum[0] += s_val[s_eleid] * matB[colid * ldb + rowid_B];
            }
            eleid++;
          } else {
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * sum[0]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * sum[0]);
            }
            T seg_rowid = binary_search_segment_number<T>(csr_row_ptr + seg_row_start, seg_row_length, eleid);
            rowid = seg_row_start + seg_rowid;
            step += csr_row_ptr[rowid + 1] - eleid;
            sum[0] = zero;
            if (ORDER_ROW) {
              sum[0] += s_val[s_eleid] * matB[rowid_B * ldb + colid];
            } else {
              sum[0] += s_val[s_eleid] * matB[colid * ldb + rowid_B];
            }
            eleid++;
          }
        }
      }
    }
    #pragma unroll
    for (int k = 0; k < factor; k++) {
      T colid = block_colid + k * warp_size + warp_laneid; 
      if (colid < N) {
        if (ORDER_ROW) {
          atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
        } else {
          atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
        }
      }
    }
  }
}

template<int block_size,
         int warp_size,
         int items_per_seg,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_merge_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ start_xs,
  const T* __restrict__ start_ys)
{
  const T items_per_block = block_size / warp_size * items_per_seg;
  __shared__ U s_val[items_per_block];
  __shared__ T s_col_ind[items_per_block];
  __shared__ T s_row_ptr[items_per_block];

  T segid = blockIdx.x * blockDim.y + threadIdx.y;
  T local_segid = threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  U *local_s_val = s_val + local_warpid * items_per_seg;
  T *local_s_col_ind = s_col_ind + local_warpid * items_per_seg;
  T *local_s_row_ptr = s_row_ptr + local_warpid * items_per_seg;

  if (segid * items_per_seg < M + nnz) {
    const T start_x = start_xs[segid];
    const T start_y = start_ys[segid];
    const T num_rows = start_xs[segid + 1] - start_x;
    const T num_nnz = start_ys[segid + 1] - start_y;

    if (warp_laneid < num_rows) {
      local_s_row_ptr[warp_laneid] = csr_row_ptr[start_x + warp_laneid + 1];
    }
    if (warp_laneid < num_nnz) {
      local_s_val[warp_laneid] = csr_val[start_y + warp_laneid];
      local_s_col_ind[warp_laneid] = csr_col_ind[start_y + warp_laneid];
    }

    T nnzid = start_y;
    T rowid = start_x;
    T local_x = 0;
    T local_y = 0;

    U col_val[factor] = {};
    W sum[factor] = {};
    int complete_current_row = 0;
    int empty_row = 1;

    for (T i = 0; i < items_per_seg && rowid < M; i++) {
      if (local_x == num_rows || nnzid < local_s_row_ptr[local_x]) {
        T rowid_B = local_s_col_ind[local_y];
        if (factor > 1) {
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                col_val[k] = matB[rowid_B * ldb + colid];
              } else {
                col_val[k] = matB[colid * ldb + rowid_B];
              }
            }
          }
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              sum[k] += local_s_val[local_y] * col_val[k];
            }
          }
        } else {
          T colid = block_colid + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              col_val[0] = matB[rowid_B * ldb + colid];
            } else {
              col_val[0] = matB[colid * ldb + rowid_B];
            }
            sum[0] += local_s_val[local_y] * col_val[0];
          }
        }
        empty_row = 0;
        local_y++;
        nnzid++;
        complete_current_row = 0;
      } else {
        if (!empty_row) {
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
              } else {
                atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
              }
            }
          }
          empty_row = 1;
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = {};
        }
        local_x++;
        rowid++;
        complete_current_row = 1;
      }
    }
    if (!complete_current_row) {
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        if (colid < N) {
          if (ORDER_ROW) {
            atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
          } else {
            atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
          }
        }
      }
    }
  }
}

template<int block_size,
         typename T>
static __global__ void
adaptive_partition_kernel(
  T M, T nnz, T items_per_seg,
  const T* __restrict__ csr_row_ptr,
  T* seg_row_id)
{
  __shared__ T csr_seg_row_ptr[block_size + 1];

  T local_tid = threadIdx.x;
  T blockid = blockIdx.x;
  T tid = blockid * blockDim.x + local_tid;
  T stride = CEIL(M, block_size);

  T csr_row_ptr_idx = MIN(((local_tid + 1) * stride), M);
  if(local_tid == 0) {
    csr_seg_row_ptr[0] = 0;
  }
  csr_seg_row_ptr[local_tid + 1] = __ldg(&csr_row_ptr[csr_row_ptr_idx]);

  __syncthreads();

  T seg_eleid = tid * items_per_seg;

  if (seg_eleid < nnz) {
    T csr_seg_row_ptr_id = binary_search_segment_number<T>(csr_seg_row_ptr, block_size, seg_eleid);

    T low = csr_seg_row_ptr_id * stride;
    T hi = MIN(low + stride, M);
    seg_row_id[tid] = low + binary_search_segment_number<T>(csr_row_ptr + low, hi - low, seg_eleid);

    if (seg_eleid + items_per_seg >= nnz) {
      seg_row_id[tid + 1] = M;
    }
  }
}

template<int block_size,
         int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_adaptive_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_row_id)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];
  __shared__ T s_seg_row_id[2];

  T blockid = blockIdx.x;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_offset = threadIdx.y * blockDim.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);
  U col_val[factor] = {};
  W sum[factor] = {};


  if (local_tid < 2) {
    s_seg_row_id[local_tid] = seg_row_id[blockid + local_tid];
  }

  __syncthreads();

  T seg_row_start = s_seg_row_id[0];
  T seg_row_end = s_seg_row_id[1];
  T seg_row_num = seg_row_end - seg_row_start;
  T row_offset = 0;

  if (seg_row_num > 0) {
    // check the first row
    T row_start = csr_row_ptr[seg_row_start];
    T row_end = csr_row_ptr[seg_row_start + 1];
    T row_length = row_end - row_start;
    T work_size = CEIL(nnz, M);
    // very long row.
    if (row_length > work_size * block_size / warp_size) {

      T rowid = seg_row_start;
      if (local_warpid == 0) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              matC[rowid * ldc + colid] *= beta;
            } else {
              matC[colid * ldc + rowid] *= beta;
            }
          }
        }
      }

      __syncthreads();

      T stride = CEIL(row_length, block_size / warp_size);
      T nnz_offset = local_warpid * stride + row_start;

      for (T i = 0; i < stride; i += warp_size) {
        if (i + warp_laneid < stride && i + warp_laneid < row_end) {
          s_val[local_tid] = csr_val[nnz_offset + i + warp_laneid];
          s_col_ind[local_tid] = csr_col_ind[nnz_offset + i + warp_laneid];
        }
        for (T j = 0; j < warp_size; j++) {
          if (i + j >= stride || i + j + nnz_offset >= row_end) break;
          T rowid_B = s_col_ind[local_offset + j];

          if (factor > 1) {
            #pragma unroll
            for (int k = 0; k < factor; k++) {
              T colid = block_colid + k * warp_size + warp_laneid; 
              if (colid < N) {
                if (ORDER_ROW) {
                  col_val[k] = matB[rowid_B * ldb + colid];
                } else {
                  col_val[k] = matB[colid * ldb + rowid_B];
                }
              }
            }

            #pragma unroll
            for (int k = 0; k < factor; k++) {
              T colid = block_colid + k * warp_size + warp_laneid; 
              if (colid < N) {
                sum[k] += s_val[local_offset + j] * col_val[k];
              }
            }
          } else {
            T colid = block_colid + warp_laneid;
            if (colid < N) {
              if (ORDER_ROW) {
                sum[0] += s_val[local_offset + j] * matB[rowid_B * ldb + colid];
              } else {
                sum[0] += s_val[local_offset + j] * matB[colid * ldb + rowid_B];
              }
            }
          }
        }
      }
      if (warp_size < PHYSICAL_WARP_SIZE) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = warpReduce<W, warp_size>(sum[k]);
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if ((local_warpid & (PHYSICAL_WARP_SIZE / warp_size - 1)) == 0) {
              if (ORDER_ROW) {
                atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
              } else {
                atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
              }
            }
          }
        }
      } else {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
            }
          }
        }
      }
      row_offset = 1;
    }
    seg_row_start += row_offset;
    seg_row_num -= row_offset;
    //can optimize.
    for (T i = 0; i < seg_row_num; i += block_size / warp_size) {
      if (i + local_warpid < seg_row_num) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
            sum[k] = {};
        }
        T rowid = seg_row_start + i + local_warpid;
        T row_start = csr_row_ptr[rowid];
        T row_end = csr_row_ptr[rowid + 1];
        T eleid = row_start;
        for (T i = row_start; i < row_end; i += warp_size) {
          eleid += warp_laneid;
          if (eleid < row_end) {
            s_val[local_tid] = csr_val[eleid];
            s_col_ind[local_tid] = csr_col_ind[eleid];
          }

          eleid -= warp_laneid;

          for (T j = 0; j < warp_size; j++) {
            if (eleid >= row_end) break;
            T rowid_B = s_col_ind[local_offset + j];

            if (factor > 1) {
              #pragma unroll
              for (int k = 0; k < factor; k++) {
                T colid = block_colid + k * warp_size + warp_laneid; 
                if (colid < N) {
                  if (ORDER_ROW) {
                    col_val[k] = matB[rowid_B * ldb + colid];
                  } else {
                    col_val[k] = matB[colid * ldb + rowid_B];
                  }
                }
              }

              #pragma unroll
              for (int k = 0; k < factor; k++) {
                T colid = block_colid + k * warp_size + warp_laneid; 
                if (colid < N) {
                  sum[k] += s_val[local_offset + j] * col_val[k];
                }
              }
            } else {
              T colid = block_colid + warp_laneid;
              if (colid < N) {
                if (ORDER_ROW) {
                  sum[0] += s_val[local_offset + j] * matB[rowid_B * ldb + colid];
                } else {
                  sum[0] += s_val[local_offset + j] * matB[colid * ldb + rowid_B];
                }
              }
            }
            eleid++;
          }
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
            } else {
              matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
            }
          }
        }
      }
    }
  }
}

template<typename T>
static __global__ void
flat_partition_kernel(
  T M, T nnz, T items_per_seg,
  const T* __restrict__ csr_row_ptr,
  T* seg_row_id)
{
  T tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < M) {
    T cur_row_seg_id = CEIL(csr_row_ptr[tid] , items_per_seg);
    T next_row_seg_id = CEIL(csr_row_ptr[tid + 1] , items_per_seg);
    if (cur_row_seg_id != next_row_seg_id) {
      for (T i = cur_row_seg_id; i < next_row_seg_id; i++) {
        seg_row_id[i] = tid;
      }
    }
  } else if (tid == M) {
    seg_row_id[CEIL(nnz, items_per_seg)] = M;
  }
}

template<int block_size,
         int warp_size,
         int work_size,
         typename T,
         typename U,
         typename V,
         typename W,
         bool ORDER_ROW>
static __global__ void
csrspmm_flat_compute_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_row_id)
{
  constexpr int block_work_size = block_size / warp_size * work_size;
  __shared__ U s_val[block_work_size];
  __shared__ T s_col_ind[block_work_size];
  __shared__ V s_sum[block_size * work_size];
  __shared__ T s_block_row_id[2];

  T blockid = blockIdx.x;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T block_eleid = blockIdx.x * blockDim.y * work_size;
  T warp_laneid = local_tid & (warp_size - 1);
  T colid = blockIdx.y * blockDim.x + warp_laneid;

  // load nnzs to LDS.
  for (T i = local_tid; i < block_work_size; i += block_size) {
    if (block_eleid + i < nnz) {
      s_val[i] = csr_val[block_eleid + i];
      s_col_ind[i] = csr_col_ind[block_eleid + i];
    }
  }

  //load seg_row_id to LDS.
  // if (local_tid < 2) {
  //   s_block_row_id[local_tid] = seg_row_id[blockid + local_tid];
  // }

  __syncthreads();

  // compute partial sum and store to LDS.
  for (T i = local_warpid; i < block_work_size; i += block_size / warp_size) {
    if (block_eleid + i < nnz && colid < N) {
      T matB_rowid = s_col_ind[i];
      if (ORDER_ROW) {
        s_sum[i * warp_size + warp_laneid] = s_val[i] * matB[matB_rowid * ldb + colid];
      } else {
        s_sum[i * warp_size + warp_laneid] = s_val[i] * matB[colid * ldb + matB_rowid];
      }
    }
  }

  __syncthreads();

  //every warp reduces partial sums from a same row.
  T block_start_row_id = seg_row_id[blockid];
  T block_end_row_id = seg_row_id[blockid + 1];
  if (block_start_row_id == block_end_row_id || csr_row_ptr[block_end_row_id] % block_work_size != 0) {
    block_end_row_id = MIN(block_end_row_id + 1, M);
  }

  for (T rowid = block_start_row_id + local_warpid; rowid < block_end_row_id; rowid += block_size / warp_size) {
    T row_start_eleid = csr_row_ptr[rowid];
    T row_end_eleid = csr_row_ptr[rowid + 1];
    T local_reduce_start_eleid = MAX(row_start_eleid, block_eleid) - block_eleid;
    T local_reduce_end_eleid = MIN(row_end_eleid, block_eleid + block_work_size) - block_eleid;
    W sum = {};
    for (T i = local_reduce_start_eleid; i< local_reduce_end_eleid; i++) {
      sum += s_sum[i * warp_size + warp_laneid];
    }
    if (colid < N) {
      if (ORDER_ROW) {
        atomicAdd(&matC[rowid * ldc + colid], alpha * sum);
      } else {
        atomicAdd(&matC[colid * ldc + rowid], alpha * sum);
      }
    }
  }
}

template<typename T, typename V, typename W>
static __global__ void
matrix_scalar_multiply_kernel(T M, T N, W alpha, V *matA) {
  T tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < M * N) {
    matA[tid] = alpha * matA[tid];
  }
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_rb_sr(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csr_row_ptr,
                const T* __restrict__ csr_col_ind,
                const U* __restrict__ csr_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  int warp_size, factor;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  }
  const int block_size = RefThreadPerBlock;
  T Mdim_worker = M;
  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Mdim_thread_per_tb = CEIL(block_size, Ndim_thread_per_tb);
  T Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    csrspmm_rowbalance_rowmajor_seqreduce_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 32 && factor == 2) {
    csrspmm_rowbalance_rowmajor_seqreduce_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 32 && factor == 1) {
    csrspmm_rowbalance_rowmajor_seqreduce_kernel<block_size, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 16 && factor == 1) {
    csrspmm_rowbalance_rowmajor_seqreduce_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 8 && factor == 1) {
    csrspmm_rowbalance_rowmajor_seqreduce_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 4 && factor == 1) {
    csrspmm_rowbalance_rowmajor_seqreduce_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_eb_sr(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csr_row_ptr,
                const T* __restrict__ csr_col_ind,
                const U* __restrict__ csr_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  int warp_size, factor, work_size;
  const int block_size = RefThreadPerBlock;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
      work_size = 32;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
      work_size = 32;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
      work_size = 32;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
      work_size = 16;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
      work_size = 8;
    } else {
      warp_size = 4;
      factor = 1;
      work_size = 4;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
      work_size = 8;
    } else {
      warp_size = 4;
      factor = 1;
      work_size = 4;
    }
  }

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T Nnzdim_worker = CEIL(nnz, (T)work_size);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  blocksPerGrid = CEIL(Nnzdim_worker, 2 * RefThreadPerBlock);
  elebalance_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    M, nnz, work_size, csr_row_ptr, seg_row_id);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    csrspmm_elebalance_rowmajor_seqreduce_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 32 && factor == 2) {
    csrspmm_elebalance_rowmajor_seqreduce_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 32 && factor == 1) {
    csrspmm_elebalance_rowmajor_seqreduce_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 16 && factor == 1) {
    csrspmm_elebalance_rowmajor_seqreduce_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 8 && factor == 1) {
    csrspmm_elebalance_rowmajor_seqreduce_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 4 && factor == 1) {
    csrspmm_elebalance_rowmajor_seqreduce_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_merge(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csr_row_ptr,
                const T* __restrict__ csr_col_ind,
                const U* __restrict__ csr_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  int warp_size, factor, work_size;
  const int block_size = RefThreadPerBlock;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
      work_size = 32;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
      work_size = 32;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
      work_size = 32;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
      work_size = 16;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
      work_size = 8;
    } else {
      warp_size = 4;
      factor = 1;
      work_size = 4;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
      work_size = 8;
    } else {
      warp_size = 4;
      factor = 1;
      work_size = 4;
    }
  }

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T merge_length = nnz + M;
  T Pathdim_worker = CEIL(merge_length, work_size);
  T *start_xs, *start_ys;
  start_xs = reinterpret_cast<T *>(externalBuffer);
  start_ys = reinterpret_cast<T *>(start_xs + Pathdim_worker + 1);

  merge_path_search_kernel<2 * RefThreadPerBlock, T><<<CEIL((Pathdim_worker + 1), 2 * RefThreadPerBlock), 2 * RefThreadPerBlock, 0, handle->stream>>>(
    start_xs,
    start_ys,
    M,
    nnz,
    merge_length,
    work_size,
    Pathdim_worker,
    csr_row_ptr);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Pathdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Pathdim_threadblock = CEIL(Pathdim_worker, Pathdim_thread_per_tb);


  dim3 gridDim(Pathdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Pathdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    csrspmm_merge_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        start_xs, start_ys);
  }
  if (warp_size == 32 && factor == 2) {
    csrspmm_merge_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        start_xs, start_ys);
  }
  if (warp_size == 32 && factor == 1) {
    csrspmm_merge_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        start_xs, start_ys);
  }
  if (warp_size == 16 && factor == 1) {
    csrspmm_merge_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        start_xs, start_ys);
  }
  if (warp_size == 8 && factor == 1) {
    csrspmm_merge_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        start_xs, start_ys);
  }
  if (warp_size == 4 && factor == 1) {
    csrspmm_merge_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        start_xs, start_ys);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_adaptive(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csr_row_ptr,
                const T* __restrict__ csr_col_ind,
                const U* __restrict__ csr_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  const int block_size = RefThreadPerBlock;
  int warp_size, factor;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  }

  const T work_size = CEIL(nnz, M);
  const T block_work_size = work_size * block_size / warp_size;
  T Mdim_worker = CEIL(nnz, work_size);
  T block_num = CEIL(nnz, block_work_size);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(block_num, 2 * RefThreadPerBlock);
  adaptive_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    M, nnz, block_work_size, csr_row_ptr, seg_row_id);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    csrspmm_adaptive_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 32 && factor == 2) {
    csrspmm_adaptive_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 32 && factor == 1) {
    csrspmm_adaptive_kernel<block_size, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 16 && factor == 1) {
    csrspmm_adaptive_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 8 && factor == 1) {
    csrspmm_adaptive_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 4 && factor == 1) {
    csrspmm_adaptive_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_flat(alphasparseHandle_t handle,
            T M, T N, T K, T nnz, W alpha, 
            const T* __restrict__ csr_row_ptr,
            const T* __restrict__ csr_col_ind,
            const U* __restrict__ csr_val,
            const U* __restrict__ matB,
            T ldb,  W beta,
            V* __restrict__ matC,
            T ldc,
            void* externalBuffer)
{
  int warp_size, work_size;
  const int block_size = RefThreadPerBlock;
  if (N > 4) {
    warp_size = 8;
    work_size = 8;
  } else {
    warp_size = 4;
    work_size = 4;
  }

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  blocksPerGrid = CEIL(M + 1, RefThreadPerBlock);
  flat_partition_kernel<T>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, nnz, block_size / warp_size * work_size, csr_row_ptr, seg_row_id);

  T Nnzdim_worker = CEIL(nnz, (T)work_size);
  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size);
  T Ndim_thread_per_tb = warp_size;
  T Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);
  if (warp_size == 8) {
    csrspmm_flat_compute_kernel<block_size, 8, 8, T, U, V, W, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 4) {
    csrspmm_flat_compute_kernel<block_size, 4, 4, T, U, V, W, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
