#include "alphasparse_spsm_common.h"

template<typename T, typename U>
__global__ static void
spts_syncfree_cuda_executor_csr_wrt_thread(const T* csrRowPtr,
                                           const T* csrColIdx,
                                           const U* csrVal,
                                           const T m,
                                           const T nrhs,
                                           const T nnz,
                                           const U alpha,
                                           U* y,
                                           T ldb)

{
  const T thread_id = threadIdx.x;
  const T stride = blockDim.x;
  const T cidx = blockIdx.x;
  extern __shared__ T get_value[];
  for (T b = cidx; b < nrhs; b += gridDim.x) {
    for (T i = thread_id; i < m; i += stride) {
      get_value[i] = 0;
    }
    __syncthreads();

    T col, j;
    U yi;
    U left_sum = {};

    for (T i = thread_id; i < m; i += stride) {
      left_sum = {};
      j = csrRowPtr[i];
      while (j < csrRowPtr[i + 1]) {
        col = csrColIdx[j];

        while (get_value[col] == 1) {
          if (col < i) {
            left_sum += csrVal[j] * y[ldb * b + col];
          } else
            break;
          j++;
          col = csrColIdx[j];
        }

        T tmp_try = !(i ^ col);
        {
          yi = alpha * y[ldb * b + i] - left_sum;
          y[ldb * b + i] = tmp_try * yi + (1 - tmp_try) * y[ldb * b + i];
          __threadfence();
          // __builtin_amdgcn_s_sleep(1);
          get_value[i] = tmp_try | get_value[i];
          __threadfence();

          if (tmp_try)
            break;
        }
      }
    }
  }
}

template<typename T, typename U, T BLOCKSIZE>
__global__ static void
trsm_opt_SM_u(const T* csrRowPtr,
            const T* csrColIdx,
            const U* csrVal,
            const T m,
            const T nrhs,
            const T nnz,
            const U alpha,
            U* y,
            T ldb,
            T* get_value)

{
  const T stride = blockDim.x * gridDim.x;
  const T cidx = blockIdx.x;
  const T thread_id = threadIdx.x + cidx * blockDim.x;

  T col, j;
  // T blockPern = (nrhs - 1) / blockDim.x + 1;
  T row = cidx % m;

  U left_sum = {};

  __shared__ T scsr_col_ind[BLOCKSIZE];
  __shared__ U scsr_val[BLOCKSIZE];

  T row_begin = csrRowPtr[row];
  T row_end = csrRowPtr[row + 1];

  // Column index into B
  T col_B = blockIdx.x / m * BLOCKSIZE + threadIdx.x;

  // Index into B (i,j)
  // T idx_B = col_B * ldb + row;
  T idx_B = row * ldb + col_B; // row major

  // Index into done array
  T id = blockIdx.x / m * m;

  // Initialize local sum with alpha and X
  U local_sum;

  if (col_B < nrhs) {
    local_sum = alpha * y[idx_B];
  } else {
    local_sum = {};
  }

  for (T j = row_begin; j < row_end; ++j) {
    // Project j onto [0, BLOCKSIZE-1]
    T k = (j - row_begin) & (BLOCKSIZE - 1);

    // Preload column indices and values into shared memory
    // This happens only once for each chunk of BLOCKSIZE elements
    if (k == 0) {
      if (threadIdx.x < row_end - j) {
        scsr_col_ind[threadIdx.x] = csrColIdx[threadIdx.x + j];
        scsr_val[threadIdx.x] = csrVal[threadIdx.x + j];
      } else {
        scsr_col_ind[threadIdx.x] = -1;
        scsr_val[threadIdx.x] = {};
        // alpha_sube(scsr_val[threadIdx.x], 1);
      }
    }

    // Wait for preload to finish
    __syncthreads();

    // Current column this lane operates on
    T local_col = scsr_col_ind[k];

    if (local_col > row) {
      break;
    }

    // Local value this lane operates with
    U local_val = scsr_val[k];

    if (local_col == row) {
      break;
    }

    // Spin loop until dependency has been resolved
    if (threadIdx.x == 0) {
      int local_done = atomicOr(&get_value[local_col + id], 0);
      unsigned int times_through = 0;
      while (!local_done) {
        local_done = atomicOr(&get_value[local_col + id], 0);
      }
    }

    // Wait for spin looping thread to finish as the whole block depends on this
    // row
    __syncthreads();

    // Make sure updated B is visible globally
    __threadfence();

    // Index into X
    // T idx_X = col_B * ldb + local_col;
    T idx_X = local_col * ldb + col_B; // row major

    // Local sum computation for each lane
    if (col_B < nrhs) {
      local_sum -= local_val * y[idx_X];
    } else {
      local_sum = {};
    }
  }

  // Store result in B
  if (col_B < nrhs) {
    y[idx_B] = local_sum;
  }

  // Wait for all threads to finish writing into global memory before we mark
  // the row "done"
  __syncthreads();

  // Make sure B is written to global memory before setting row is done flag
  __threadfence();

  if (threadIdx.x == 0) {
    // Write the "row is done" flag
    atomicOr(&get_value[row + id], 1);
  }
}

template<typename T, typename U>
alphasparseStatus_t
spsm_csr_u_lo(alphasparseHandle_t handle,
              T m,
              T nrhs,
              T nnz,
              const U alpha,
              const U* csr_val,
              const T* csr_row_ptr,
              const T* csr_col_ind,
              U* B,
              T ldb,
              U* C)
{
  T ldimB = nrhs;
  U* Bt = NULL;
  cudaMalloc((void**)&Bt, sizeof(U) * m * ldimB);

  {
#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
    dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
    dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);
    csrsm_transpose<CSRSM_DIM_X, CSRSM_DIM_Y, T, U>
      <<<csrsm_blocks, csrsm_threads, 0, handle->stream>>>(
        m, nrhs, B, ldb, Bt, ldimB);

#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
  }

  const T blockdim = 1024;
  const T griddim = ((nrhs - 1) / blockdim + 1) * m;
  const T narrays = (nrhs - 1) / blockdim + 1;
  T* get_value;

  cudaMalloc((void**)&get_value, sizeof(T) * m * narrays);

  cudaMemsetAsync(get_value, 0, sizeof(T) * m * narrays, handle->stream);
  trsm_opt_SM_u<T, U, blockdim>
    <<<dim3(griddim), dim3(blockdim), 0, handle->stream>>>(csr_row_ptr,
                                                           csr_col_ind,
                                                           csr_val,
                                                           m,
                                                           nrhs,
                                                           nnz,
                                                           alpha,
                                                           Bt,
                                                           ldimB,
                                                           get_value);
  {
#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
    dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
    dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);
    csrsm_transpose_back<CSRSM_DIM_X, CSRSM_DIM_Y, T, U>
      <<<csrsm_blocks, csrsm_threads, 0, handle->stream>>>(
        m, nrhs, Bt, ldimB, B, ldb);
#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
  }
  cudaMemcpy(C, B, m * nrhs * sizeof(U), cudaMemcpyDeviceToDevice);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
