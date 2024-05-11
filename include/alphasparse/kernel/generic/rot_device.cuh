template <typename T, typename U>
__global__ static void
rot_device(T nnz, U *x_val, const T *x_ind, U *y, U c, U s);
