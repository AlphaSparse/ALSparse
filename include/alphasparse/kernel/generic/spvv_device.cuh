// #include <cuda_runtime.h>
// #include <cuda_fp16.h>

template<typename T, typename U>
__global__ static void
spvv_device(T nnz, const U* x_val, const T* x_ind, const U* y, U* result);

template<typename U>
__global__ static void
reduce_result(int blockPerGrid, U* dev_part_c, U* dev_result);
