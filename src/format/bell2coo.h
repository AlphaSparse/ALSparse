#pragma once

#include "alphasparse.h"
#include "coo_order.h"
#include "alphasparse_create_bell.h"
#include "transpose_csc.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>

template <typename T, typename U>
alphasparseStatus_t
alphasparseBell2coo(alphasparseSpMatDescr_t &bell, alphasparseSpMatDescr_t &coo)
{
  int m = bell->rows;
  int n = bell->cols;
  int blocksize = bell->block_dim;
  int mb = (m + blocksize - 1) / blocksize;
  int nb = (n + blocksize - 1) / blocksize;
  int ell_cols = bell->ell_cols;
  int max_block_cols = ell_cols / blocksize;
  int bell_values_num = m * ell_cols;
  int *bell_col_index = (int *)malloc(mb * max_block_cols * sizeof(int));
  U *bell_values = (U *)malloc(bell_values_num * sizeof(U));
  cudaMemcpy(bell_col_index,
             bell->col_data,
             mb * max_block_cols * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(bell_values,
             bell->val_data,
             bell_values_num * sizeof(U),
             cudaMemcpyDeviceToHost);
  std::vector<int> coo_row_index;
  std::vector<int> coo_col_index;
  std::vector<U> coo_values;
  int block_row_index, block_col_index;
  U bell_value;
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < ell_cols; j++)
    {
      bell_value = bell_values[i * ell_cols + j];
      if (!is_zero(bell_value))
      {
        coo_values.push_back(bell_value);
        coo_row_index.push_back(i);
        block_row_index = i / blocksize;
        block_col_index = j / blocksize;
        coo_col_index.push_back(
            bell_col_index[block_row_index * max_block_cols + block_col_index] *
                blocksize +
            j % blocksize);
      }
    }
  }
  int nnz = coo_row_index.size();
  coo_order<int32_t, U>(
      nnz, coo_row_index.data(), coo_col_index.data(), coo_values.data());
  int *d_coo_row_index = NULL;
  int *d_coo_col_index = NULL;
  U *d_coo_values = NULL;
  cudaMalloc((void **)&d_coo_row_index, nnz * sizeof(int));
  cudaMalloc((void **)&d_coo_col_index, nnz * sizeof(int));
  cudaMalloc((void **)&d_coo_values, nnz * sizeof(U));
  cudaMemcpy(d_coo_row_index,
             coo_row_index.data(),
             nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_col_index,
             coo_col_index.data(),
             nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(
      d_coo_values, coo_values.data(), nnz * sizeof(U), cudaMemcpyHostToDevice);
  alphasparseCreateCoo(&coo,
                       m,
                       n,
                       nnz,
                       d_coo_row_index,
                       d_coo_col_index,
                       d_coo_values,
                       bell->row_type,
                       bell->idx_base,
                       bell->data_type);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
