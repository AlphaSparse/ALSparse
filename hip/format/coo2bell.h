#pragma once

#include "alphasparse.h"
#include "alphasparse_create_bell.h"
#include "transpose_csc.h"
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <vector>
#include <algorithm>

template <typename T, typename U>
alphasparseStatus_t
alphasparseCoo2bell(alphasparseSpMatDescr_t &coo,
                    alphasparseSpMatDescr_t &bell,
                    int blocksize)
{
  int m = coo->rows;
  int n = coo->cols;
  int nnz = coo->nnz;
  int mb = (m + blocksize - 1) / blocksize;
  int nb = (n + blocksize - 1) / blocksize;
  int *row_index = (int *)malloc(nnz * sizeof(int));
  int *col_index = (int *)malloc(nnz * sizeof(int));
  U *values = (U *)malloc(nnz * sizeof(U));
  hipMemcpy(
      row_index, coo->row_data, nnz * sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(
      col_index, coo->col_data, nnz * sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(values, coo->val_data, nnz * sizeof(U), hipMemcpyDeviceToHost);
  bool *block_mat_flag = (bool *)calloc(mb * nb, sizeof(bool));
  for (int i = 0; i < nnz; i++)
  {
    block_mat_flag[row_index[i] / blocksize * nb + col_index[i] / blocksize] =
        true;
  }
  size_t max_block_cols = 0, block_cols;
  std::vector<std::vector<int>> ell_block_col_indexes(mb);
  for (int i = 0; i < mb; i++)
  {
    for (int j = 0; j < nb; j++)
    {
      if (block_mat_flag[i * nb + j])
      {
        ell_block_col_indexes[i].push_back(j);
      }
    }
    block_cols = ell_block_col_indexes[i].size();
    if (block_cols > max_block_cols)
      max_block_cols = block_cols;
  }
  int ell_cols = max_block_cols * blocksize;
  U *ell_values = (U *)calloc(m * ell_cols, sizeof(U));
  int block_row_index, block_col_index;
  int block_row_index_in, block_col_index_in;
  std::vector<int> ell_block_col_index;
  ptrdiff_t pos;
  for (int i = 0; i < nnz; i++)
  {
    block_row_index = row_index[i] / blocksize;
    block_col_index = col_index[i] / blocksize;
    ell_block_col_index = ell_block_col_indexes[block_row_index];
    pos = std::find(ell_block_col_index.begin(),
                    ell_block_col_index.end(),
                    block_col_index) -
          ell_block_col_index.begin();
    if (pos < ell_block_col_index.size())
    {
      block_row_index_in = row_index[i] % blocksize;
      block_col_index_in = col_index[i] % blocksize;
      ell_values[(block_row_index * blocksize + block_row_index_in) * ell_cols +
                 pos * blocksize + block_col_index_in] = values[i];
    }
  }
  int *ell_col_inx = (int *)malloc(mb * max_block_cols * sizeof(int));
  memset(ell_col_inx, -1, mb * max_block_cols * sizeof(int));
  for (int i = 0; i < mb; i++)
  {
    ell_block_col_index = ell_block_col_indexes[i];
    for (int j = 0; j < ell_block_col_index.size(); j++)
    {
      ell_col_inx[i * max_block_cols + j] = ell_block_col_index[j];
    }
  }

  int *d_ell_col_inx = NULL;
  hipMalloc((void **)&d_ell_col_inx, mb * max_block_cols * sizeof(int));
  hipMemcpy(d_ell_col_inx,
             ell_col_inx,
             mb * max_block_cols * sizeof(int),
             hipMemcpyHostToDevice);
  U *d_ell_values = NULL;
  hipMalloc((void **)&d_ell_values, m * ell_cols * sizeof(U));
  hipMemcpy(
      d_ell_values, ell_values, m * ell_cols * sizeof(U), hipMemcpyHostToDevice);
  alphasparseCreateBlockedEll(&bell,
                              m,
                              n,
                              blocksize,
                              ell_cols,
                              d_ell_col_inx,
                              d_ell_values,
                              coo->row_type,
                              coo->idx_base,
                              coo->data_type);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
