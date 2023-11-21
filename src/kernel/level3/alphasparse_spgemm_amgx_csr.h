#include "alphasparse.h"
#include "csrspgemm_device_amgx.h"
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             cudaGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

template <typename T, typename U>
alphasparseStatus_t spgemm_csr_amgx(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const U alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const U beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    const int m_grid_size = 1024, m_max_warp_count = 8;
    const int NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;
    const int m_num_threads_per_row_count = 32;
    const int m_num_threads_per_row_compute = 32;
    
    int * m_keys = NULL;
    U * m_vals = NULL;
    
    int m_gmem_size = 512 * 2;
    // size_t t = 0;
    size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(int);
    CHECK_CUDA(cudaMalloc( (void **) &m_keys, sz ));
    // t += sz;
    sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(U);
    CHECK_CUDA(cudaMalloc( (void **) &m_vals, sz ));  
    cudaMemset(matC->row_data, 0, (matC->rows+1) * sizeof(T));
    // t += sz;
    // printf("buffer size %d\n", t);
    // T * m_seq_offsets = NULL;
    // CHECK_CUDA(cudaMalloc( (void **) &m_seq_offsets, matA->rows * sizeof(T) ));
    // thrust::sequence(thrust::device, m_seq_offsets, m_seq_offsets + matA->rows);

    count_non_zeroes<T>(matA, matB, matC, m_num_threads_per_row_count, m_keys, NULL, NULL, NULL, NULL);

    CHECK_CUDA( cudaGetLastError() );
    // Compute row offsets.
    thrust::exclusive_scan(thrust::device, matC->row_data, matC->row_data + (matC->rows+1), matC->row_data );

    CHECK_CUDA( cudaGetLastError() );
    // Allocate memory to store columns/values.
    int num_vals = 0;
    cudaMemcpy( &num_vals, &matC->row_data[matC->rows], sizeof(int), cudaMemcpyDeviceToHost );
    // printf("nnz C %d\n", num_vals);
    if(matC->nnz != num_vals)
    {
      matC->nnz = num_vals;
      if(matC->col_data == nullptr)
        CHECK_CUDA( cudaMalloc((void **)&matC->col_data, sizeof(T) * num_vals));
      if(matC->val_data == nullptr)
        CHECK_CUDA( cudaMalloc((void **)&matC->val_data, sizeof(U) * num_vals));
    }

    bool done = false;

    if ( m_num_threads_per_row_count != m_num_threads_per_row_compute )
    {
        // Reset the status.
        int status = 0;
        // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
        // properly set but it is responsible for setting the work queue.
        status = compute_values<T,U>( matA, matB, matC, m_num_threads_per_row_compute, m_keys, m_vals, alpha, NULL, NULL, NULL, NULL );
        // Read the result from count_non_zeroes.
        done = status == 0;
    }
    // Re-run if needed.
    if ( !done )
    {
        compute_values<T,U>( matA, matB, matC, m_num_threads_per_row_count, m_keys, m_vals, alpha, NULL, NULL, NULL, NULL);
    }
    CHECK_CUDA(cudaFree(m_keys));
    CHECK_CUDA(cudaFree(m_vals));
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
