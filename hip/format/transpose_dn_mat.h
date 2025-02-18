#include "hip/hip_runtime.h"
#include "alphasparse.h"

template<typename T, typename U>
__global__ void
transpose_dn_mat_device(const T rows, const T cols, const U* in, U* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    int index_in = i * cols + j;
    int index_out = j * rows + i;
    out[index_out] = in[index_in];
  }
}

template<typename T, typename U>
alphasparseStatus_t
transpose_dn_mat(alphasparseHandle_t handle,
        const T m,
        const T n,
        const U* mat_row_major,
        U* mat_col_major)
{
  // const int threadPerBlock = 256;
  // const T blockPerGrid =
  //   min((T)2, ((T)threadPerBlock + m * n - (T)1) / threadPerBlock);
  dim3 threadPerBlock(32, 32);
  dim3 blockPerGrid((m + threadPerBlock.x - (T)1) / threadPerBlock.x,
                    (n + threadPerBlock.y - (T)1) / threadPerBlock.y);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_dn_mat_device<T, U>), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, 
      m, n, mat_row_major, mat_col_major);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}