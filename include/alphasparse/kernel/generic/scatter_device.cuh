template <typename T, typename U>
__global__ static void
scatter_device(const T nnz, const U *x_val, const T *x_ind, U *y);
