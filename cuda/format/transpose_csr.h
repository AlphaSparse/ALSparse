#pragma once

#include "alphasparse.h"
#include "alphasparse_create_csr.h"
#include <memory.h>
#include <stdlib.h>
#include <vector>

template <typename T, typename U>
alphasparseStatus_t
transpose_csr(alphasparseSpMatDescr_t &A)
{
  // 计算矩阵的转置
  int num_row = A->rows;
  int num_col = A->cols;
  int num_nonzeros = A->nnz;
  U *Avalues = (U *)malloc(sizeof(U) * A->nnz);
  int *Arows_offset = (int *)malloc(sizeof(int) * (A->rows + 1));
  int *Acol_indx = (int *)malloc(sizeof(int) * A->nnz);
  cudaMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), cudaMemcpyDeviceToHost);
  cudaMemcpy(Arows_offset,
             A->row_data,
             (A->rows + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(
      Acol_indx, A->col_data, A->nnz * sizeof(int), cudaMemcpyDeviceToHost);
  // 统计每一列包含的非零元素的数量
  std::vector<int> column_counts(num_col, 0);
  for (int i = 0; i < num_nonzeros; i++)
  {
    column_counts[Acol_indx[i]]++;
  }
  // 计算转置后每一行元素在values和row_ids中的起始位置
  int *rows_offset = (int *)alpha_memalign((uint64_t)(num_col + 1) * sizeof(int),
                                           DEFAULT_ALIGNMENT);
  memset(rows_offset, 0, sizeof(int) * (num_col + 1));
  for (int i = 1; i <= num_col; i++)
  {
    rows_offset[i] = rows_offset[i - 1] + column_counts[i - 1];
  }
  U *values =
      (U *)alpha_memalign((uint64_t)A->nnz * sizeof(U), DEFAULT_ALIGNMENT);
  int *col_indx =
      (int *)alpha_memalign((uint64_t)A->nnz * sizeof(int), DEFAULT_ALIGNMENT);
  memset(values, 0, sizeof(U) * A->nnz);
  memset(col_indx, 0, sizeof(int) * A->nnz);

  // 将非零元素按列号放入转置后的矩阵中
  for (int i = 0; i < num_row; i++)
  {
    for (int j = Arows_offset[i]; j < Arows_offset[i + 1]; j++)
    {
      int col_id = Acol_indx[j];
      int dest_index = rows_offset[col_id];
      col_indx[dest_index] = i;
      values[dest_index] = Avalues[j];
      // rows_offset[col_id]++;
    }
  }

  // 恢复每一行元素在values和row_ids中的起始位置
  // for (int i = num_col; i >= 1; i--)
  // {
  //   rows_offset[i] = rows_offset[i - 1];
  // }
  // rows_offset[0] = 0;
  int *drows_offset = NULL;
  CHECK_CUDA(cudaMalloc((void **)&drows_offset, (num_col + 1) * sizeof(int)));
  cudaMemcpy(drows_offset,
             rows_offset,
             (num_col + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  A->row_data = drows_offset;
  cudaMemcpy(A->val_data, values, A->nnz * sizeof(U), cudaMemcpyHostToDevice);
  cudaMemcpy(
      A->col_data, col_indx, A->nnz * sizeof(int), cudaMemcpyHostToDevice);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
