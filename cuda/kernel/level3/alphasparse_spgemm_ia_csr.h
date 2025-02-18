#include <iostream>

#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>

#include <cusp/sort.h>
#include <cusp/format_utils.h>

template <typename T>
__global__ void csr_to_coo(T row, T* row_data, T* A_coo)
{
    for(T i=0;i<row;i++)
    {
      for(T j=row_data[i];j<row_data[i+1];j++)
      A_coo[j]=i;
    }
}

template <typename T>
__global__ void phase1(T nnz, T* segment_lengths, T* output_ptr)
{
    output_ptr[nnz] = output_ptr[nnz - 1] + segment_lengths[nnz - 1];
}

template <typename T, typename U>
alphasparseStatus_t spgemm_csr_ia(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const U alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const U beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    matC->row=matA->row;
    matC->col=matB->col;

	  T* A_coo;
    cudaMalloc((void**)&A_coo,sizeof(T)*(matA->row));
    csr_to_coo<<<1,1>>>(matA->row,matA->row_data,A_coo);

    T* B_row_length;
    cudaMalloc((void**)&B_row_length,sizeof(T)*(matB->row));
    thrust::transform(thrust::device, &matB->row_data[1], &matB->row_data[matB->row+1], &matB->row_data[0], &B_row_length[0], thrust::minus<T>());

    T* segment_lengths;
    cudaMalloc((void**)&segment_lengths,sizeof(T)*(matA->nnz));
    thrust::gather(thrust::device, matA->col_data, &matA->col_data[matA->nnz], B_row_length, segment_lengths);

    T* output_ptr;
    cudaMalloc((void**)&output_ptr,sizeof(T)*(matA->nnz+1));
    thrust::exclusive_scan(thrust::device, segment_lengths, &segment_lengths[matA->nnz], output_ptr, T(0));

    phase1<<<1,1>>>(matA->nnz,segment_lengths,output_ptr);

    T coo_num_nonzeros;
    cudaMemcpy(&coo_num_nonzeros, &output_ptr[matA->nnz], sizeof(T), cudaMemcpyDeviceToHost);

	  T workspace_capacity = thrust::min<T>(coo_num_nonzeros, 16 << 20);

    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        // divide free bytes by the size of each workspace unit
        T max_workspace_capacity = free / (4 * sizeof(T) + sizeof(U));

        // use at most one third of the remaining capacity
        workspace_capacity = thrust::min<T>(max_workspace_capacity / 3, workspace_capacity);
    }

	  T* A_gather_locations;
    T* B_gather_locations;
    T* I;
    T* J;
    U* V;

		T begin_row      = 0;
    T end_row        = matA->row;
    T begin_segment  = 0;
    T end_segment    = matA->nnz;
    T workspace_size = coo_num_nonzeros;

    cudaMalloc((void**)&A_gather_locations,sizeof(T)*(workspace_size));
    cudaMalloc((void**)&B_gather_locations,sizeof(T)*(workspace_size));
    cudaMalloc((void**)&I,sizeof(T)*(workspace_size));
    cudaMalloc((void**)&J,sizeof(T)*(workspace_size));
    cudaMalloc((void**)&V,sizeof(U)*(workspace_size));

    thrust::fill(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], 0);

    thrust::scatter_if(thrust::device, thrust::counting_iterator<T>(begin_segment), thrust::counting_iterator<T>(end_segment), output_ptr, segment_lengths, A_gather_locations);

    thrust::inclusive_scan(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], A_gather_locations, thrust::maximum<T>());

    thrust::fill(thrust::device, B_gather_locations, &B_gather_locations[workspace_size], 1);

	  thrust::scatter_if(thrust::device, thrust::make_permutation_iterator(matB->row_data, matA->col_data), thrust::make_permutation_iterator(matB->row_data, matA->col_data) + end_segment, output_ptr, segment_lengths, B_gather_locations);

    thrust::inclusive_scan_by_key(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], B_gather_locations, B_gather_locations);


	  thrust::gather(thrust::device,
                   A_gather_locations, &A_gather_locations[workspace_size],
                   A_coo,
                   I);
	  thrust::gather(thrust::device,
                   B_gather_locations, &B_gather_locations[workspace_size],
                   matB->col_data,
                   J);

    thrust::transform(thrust::device, thrust::make_permutation_iterator(matA->val_data, A_gather_locations), thrust::make_permutation_iterator(matA->val_data, &A_gather_locations[workspace_size]), thrust::make_permutation_iterator(matB->val_data, B_gather_locations), V, thrust::multiplies<U>());

    cusp::array1d_view< T* > II(I,&I[workspace_size]);
    cusp::array1d_view< T* > JJ(J,&J[workspace_size]);
    cusp::array1d_view< U* > VV(V,&V[workspace_size]);

    cusp::sort_by_row_and_column(thrust::device, II, JJ, VV);

	  T NNZ = thrust::inner_product(thrust::device,
                                          thrust::make_zip_iterator(thrust::make_tuple(II.begin(), JJ.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(II.end (),  JJ.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(II.begin(), JJ.begin())) + 1,
                                          T(0),
                                          thrust::plus<T>(),
                                          thrust::not_equal_to< thrust::tuple<T,T> >()) + 1;

    matC->nnz=NNZ;

    cudaMalloc((void**)&matC->row_data,sizeof(T)*(matA->row+1));
    cudaMalloc((void**)&matC->col_data,sizeof(T)*(NNZ));
    cudaMalloc((void**)&matC->val_data,sizeof(U)*(NNZ));

	  thrust::reduce_by_key
    (thrust::device,
     thrust::make_zip_iterator(thrust::make_tuple(I, J)),
     thrust::make_zip_iterator(thrust::make_tuple(&I[workspace_size], &J[workspace_size])),
     V,
     thrust::make_zip_iterator(thrust::make_tuple(matC->row_data, matC->col_data)),
     matC->val_data,
     thrust::equal_to< thrust::tuple<T,T> >(),
     thrust::plus<U>() );

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
