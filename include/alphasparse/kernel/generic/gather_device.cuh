template <typename T, typename U>
__global__ static void
gather_device(const T nnz, const U *y, U *x_val, const T *x_ind);
