#include "alphasparse_spsm_common.h"

template<typename T, typename U, T BLOCKSIZE>
__global__ static void
trsm_opt_upper(const T* csrRowPtr,
               const T* csrColIdx,
               const U* csrVal,
               const T m,
               const T nrhs,
               const T nnz,
               const U alpha,
               const U* diag,
               U* y,
               T ldb,
               T* get_value)

{
  const T stride = blockDim.x * gridDim.x;
  const T cidx = blockIdx.x;
  const T thread_id = threadIdx.x + cidx * blockDim.x;

  T col, j;
  // T blockPern = (nrhs - 1) / hipBlockDim_x + 1;
  T row = m - 1 - cidx % m;
  T sub;

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

  for (T j = row_end - 1; j >= row_begin; --j) {
    // Project j onto [0, BLOCKSIZE-1]
    T k = (row_end - j - 1) & (BLOCKSIZE - 1);

    // Preload column indices and values into shared memory
    // This happens only once for each chunk of BLOCKSIZE elements
    if (k == 0) {
      if ((T)threadIdx.x + j - BLOCKSIZE + 1 >= row_begin) {
        // assert(threadIdx.x + j - BLOCKSIZE + 1 < row_end);
        scsr_col_ind[threadIdx.x] = csrColIdx[threadIdx.x + j - BLOCKSIZE + 1];
        scsr_val[threadIdx.x] = csrVal[threadIdx.x + j - BLOCKSIZE + 1];
      } else {
        scsr_col_ind[threadIdx.x] = -1;
        scsr_val[threadIdx.x] = {};
      }
      sub = BLOCKSIZE < j - row_begin + 1 ? BLOCKSIZE : j - row_begin + 1;
      sub = BLOCKSIZE - sub;
    }

    // Wait for preload to finish
    __syncthreads();

    // k += sub;
    k = BLOCKSIZE - (row_end - j);

    // Current column this lane operates on
    T local_col = scsr_col_ind[k];

    // if(cidx == 4999)
    //     printf("tid %u k %d indx %d round %d\n", threadIdx.x, k,
    //     scsr_col_ind[k], row_end - j);

    __syncthreads();

    if (local_col < row) {
      continue;
    }
    if (local_col == row) {
      break;
    }

    // // Local value this lane operates with
    U local_val = scsr_val[k];
    // // Spin loop until dependency has been resolved
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
    y[idx_B] = local_sum / diag[row];
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
spsm_csr_n_hi(alphasparseHandle_t handle,
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
  U* diag;
  cudaMalloc((void**)&diag, sizeof(U) * m);
  cudaMemset(diag, '\0', sizeof(U) * m);

  const T threadPerBlock = 256;
  const int blockPerGrid =
    min((T)32, (threadPerBlock + nrhs - 1) / threadPerBlock);

  // todo diag的计算是否应合并
  get_diags<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(
    m, csr_val, csr_row_ptr, csr_col_ind, diag);

  // hipLaunchKernelGGL(trsm_plain, dim3(blockPerGrid), dim3(threadPerBlock), 0,
  // handle->stream,
  //                    m, nrhs, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind,
  //                    B, ldb, diag);

  const T blockdim = 1024;

  // Leading dimension for transposed B
  const T ldimB = nrhs;
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

  const T griddim = ((nrhs - 1) / blockdim + 1) * m;
  const T narrays = (nrhs - 1) / blockdim + 1;
  T* get_value;

  cudaMalloc((void**)&get_value, sizeof(T) * m * narrays);

  cudaMemsetAsync(get_value, 0, sizeof(T) * m * narrays, handle->stream);
    trsm_opt_upper<T, U, blockdim>
    <<<dim3(griddim), dim3(blockdim), 0, handle->stream>>>(csr_row_ptr,
                                                           csr_col_ind,
                                                           csr_val,
                                                           m,
                                                           nrhs,
                                                           nnz,
                                                           alpha,
                                                           diag,
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
