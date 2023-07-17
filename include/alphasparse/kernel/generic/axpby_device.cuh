template <typename T, typename U, typename V>
__global__ static void
axpby_device(const T size, const T nnz, const V alpha, const U *x_val, const V beta, const T *x_ind, U *y);
