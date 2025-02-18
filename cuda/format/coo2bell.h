#pragma once

#include "alphasparse.h"
#include "alphasparse_create_bell.h"
#include "transpose_csc.h"
#include "coo2csr.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define warpSize 32
typedef struct ListNode {
    int data;
    struct ListNode* next;
    ListNode(int x) : data(x), next(NULL) {}
} iNode;
// Function to insert a new node into the linked list in ascending order
void insertNode(iNode** head, int value) {
    iNode* newNode = (iNode*)malloc(sizeof(iNode));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed for a new node\n");
        exit(1);
    }
    newNode->data = value;
    newNode->next = NULL;

    iNode* current = *head;
    iNode* prev = NULL;

    while (current != NULL && current->data < value) {
        prev = current;
        current = current->next;
    }

    if (prev == NULL) {
        // Insert at the beginning
        newNode->next = *head;
        *head = newNode;
    } else {
        // Insert in the middle or at the end
        prev->next = newNode;
        newNode->next = current;
    }
}

template <typename T>
T findMaxNnz(T *rowPtr, T *colIndex, T num_rows, int block_size) {

    T max = 0;
    int num_blocks = num_rows / block_size;
    if (num_rows % block_size != 0)
        num_blocks++;

    for(int i=0; i < num_rows; i++) {
        int flag=0;
		int number_of_cols = 0;
		for(int j=rowPtr[i]; j<rowPtr[i+1]; j++) {
            if (flag <= colIndex[j]) {
                flag = (colIndex[j]/block_size) * block_size + block_size;
                number_of_cols++;
            }
        }
        if (number_of_cols > max)
            max = number_of_cols;
	}

    return max*block_size;
}

template <typename T>
/* Creates the array of block indexes for the blocked ell format */
T *createBlockIndex(T *rowPtr, T *colIndex, T num_rows, int block_size, T ell_cols) {

    long int mb = num_rows/block_size, nb = ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;

    T *hA_columns = (T *)calloc(nb * mb, sizeof(T));
    int ctr = 0;

    memset(hA_columns, -1, nb * mb * sizeof(T));

    for(int i=0; i<mb; i++) {

        int *flag = (int *)calloc(mb, sizeof(int));
        iNode* block_list = NULL;

        for (int j = 0; j < block_size; j++) {
            int id = block_size*i + j;
            int index = 0;
            if (id >= num_rows)
                break;

            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++) {    
                index = (colIndex[k]/block_size);
                if (flag[index] == 0) {
                    insertNode(&block_list, index);
                    flag[index] = 1;
                }
            }
        }
        
        while (block_list != NULL) {
            iNode *temp = block_list;
            hA_columns[ctr++] = block_list->data;
            block_list = block_list->next;
            free(temp);
        }
        ctr = i*nb+nb;
        free(flag);
    }

    return hA_columns; 
}

template <typename T, typename U>
/* Creates the array of values for the blocked ell format */
U *createValueIndex(T *rowPtr, T *colIndex, U *values, T *hA_columns, T num_rows, int block_size, T ell_cols) {

    /* Allocate enough memory for the array */
    U *hA_values = (U *)calloc(num_rows * ell_cols, sizeof(U));
    long int mb = num_rows/block_size, nb = ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;

    /* Set all values to 0 */
    memset(hA_values, 0, num_rows * ell_cols * sizeof(U));

    /* Iterate the blocks in the y axis */
    for (int i=0; i<mb;i++){

        /* Iterate the lines of each block */
        for (int l = 0; l<block_size; l++) {
            int ctr = 0;

            /* Iterate the blocks in the block_id array (x axis) */
            for (int j = 0; j < nb; j++) {
                int id = nb*i + j;
                if (hA_columns[id] == -1)
                    break;

                /* Iterate each line of the matrix */
                for(int k=rowPtr[i*block_size+l]; k<rowPtr[i*block_size+l+1]; k++) {  

                    /* If the element is not in the same block, skip*/
                    if (colIndex[k]/block_size > hA_columns[id])
                        break;
                    else if (colIndex[k]/block_size == hA_columns[id]) 
                        hA_values[i*ell_cols*block_size+l*ell_cols+j*block_size+(colIndex[k]-(hA_columns[id]*block_size))] = values[k];
                }
            }
        }
    }
    
    return hA_values;
}

__global__ void computeMaxNnzPerRowPerBlock(const int m, const int total_warps, const int *__restrict__ rows,
                                            uint32_t *__restrict__ max_nnz_per_row_per_warp,
                                            int *__restrict__ row_warp_mapping, int *__restrict__ row_in_warp) {
  const int row = threadIdx.x + blockDim.x * blockIdx.x;
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int warps_per_block = blockDim.x / warpSize;
  const int warp_global = warp + warps_per_block * blockIdx.x;

  int nnz = 0;
  if (row < m) {
    nnz = rows[row + 1] - rows[row];
    row_warp_mapping[row] = warp_global;
    row_in_warp[row] = row % warpSize;
  }

  /* reduce this value across the warp */
#pragma unroll
  for (int i = warpSize >> 1; i > 0; i >>= 1)
    nnz = max(__shfl_down_sync(0xffffffff, nnz, i, warpSize), nnz);

  if (warp_global < total_warps) {
    if (lane == 0) {
     max_nnz_per_row_per_warp[warp_global] = (uint32_t)nnz;
    }
  }
}

template <int THREADS_PER_ROW, typename U>
__global__ void
fillBlockColEll(const int m, const int total_warps,
                uint32_t *__restrict__ max_nnz_per_row_per_warp,
                const int *__restrict__ row_warp_mapping,
                const int *__restrict__ row_in_warp,
                const int *__restrict__ rows, const int *__restrict__ cols,
                const U *__restrict__ vals, uint32_t *__restrict__ ell_cols,
                U *__restrict__ ell_vals) {

  const int lane = threadIdx.x % THREADS_PER_ROW;
  const int row = threadIdx.x / THREADS_PER_ROW +
                  (blockDim.x / THREADS_PER_ROW) * blockIdx.x;

  int warp = total_warps - 1, rowInWarp = 0;
  if (row < m) {
    warp = row_warp_mapping[row];
    rowInWarp = row_in_warp[row];
  }
  uint32_t start = max_nnz_per_row_per_warp[warp];
  uint32_t end = max_nnz_per_row_per_warp[warp + 1];

  uint32_t max_nnz_in_warp = end - start;
  start *= warpSize;
  end *= warpSize;

  if (row < m) {
    /* determine the start and ends of each row */
    int r[2];
    if (lane < 2)
      r[lane] = rows[row + lane];
    r[0] = __shfl_sync(0xFFFFFFF,r[0], 0, THREADS_PER_ROW);
    r[1] = __shfl_sync(0xFFFFFFF,r[1], 1, THREADS_PER_ROW);
    int maxcol = -1;

    for (int i = r[0] + lane; i < r[1]; i += THREADS_PER_ROW) {
      int col = cols[i];
      U val = vals[i];
      int write_index = start + (i - r[0]) * warpSize + rowInWarp;
      ell_cols[write_index] = (uint32_t)col;
      ell_vals[write_index] = val;
      if (col > maxcol)
        maxcol = col;
    }

    for (int i = THREADS_PER_ROW >> 1; i > 0; i >>= 1)
      maxcol = max(maxcol, __shfl_down_sync(0xffffffff, maxcol, i, THREADS_PER_ROW));
    maxcol = __shfl_sync(0xFFFFFFF,maxcol, 0, THREADS_PER_ROW);

    for (int i = r[1] - r[0] + lane; i < max_nnz_in_warp;
         i += THREADS_PER_ROW) {
      int write_index = start + i * warpSize + rowInWarp;
      ell_cols[write_index] = 0;
      ell_vals[write_index] = U{};
    }
  }
}

unsigned long prevPowerOf2(unsigned long v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}

template <typename T, typename U>
alphasparseStatus_t
alphasparseCoo2bell(alphasparseSpMatDescr_t &coo,
                    alphasparseSpMatDescr_t &bell,
                    int blocksize)
{
  int m = coo->rows;
  int n = coo->cols;
  int nnz = coo->nnz;
  printf("converting COO to BELL with size %d\n", blocksize);
  int *row_index = (int *)malloc(nnz * sizeof(int));
  int *col_index = (int *)malloc(nnz * sizeof(int));
  int *row_ptr = NULL;
  cudaMalloc((void **)&row_ptr, (m+1) * sizeof(int));
  U *values = (U *)malloc(nnz * sizeof(U));
  cudaMemcpy(
      row_index, coo->row_data, nnz * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      col_index, coo->col_data, nnz * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(values, coo->val_data, nnz * sizeof(U), cudaMemcpyDeviceToHost);

  int *csrRowPtr = (int*)calloc((m+1), sizeof(int));
  alphasparseXcoo2csr((const int *)coo->row_data, nnz, m, row_ptr);
  PRINT_IF_CUDA_ERROR(cudaMemcpy(csrRowPtr, row_ptr, (m+1) * sizeof(int), cudaMemcpyDeviceToHost));

  T A_ell_cols = findMaxNnz(csrRowPtr, col_index, m, blocksize);
  printf("A_ell_cols %d\n",A_ell_cols);
  T *hA_columns = createBlockIndex(csrRowPtr, col_index, m, blocksize, A_ell_cols);
  U *hA_values = createValueIndex(csrRowPtr, col_index, values, hA_columns, m, blocksize, A_ell_cols);
  T nb = A_ell_cols / blocksize;
  T *d_ell_col_inx = NULL;
  T mb = m / blocksize;
  long bs_squre = mb * nb;
  cudaMalloc((void **)&d_ell_col_inx, bs_squre * sizeof(T));
  cudaMemcpy(d_ell_col_inx,
             hA_columns,
             bs_squre * sizeof(T),
             cudaMemcpyHostToDevice);
  U *d_ell_values = NULL;
  cudaMalloc((void **)&d_ell_values, m * A_ell_cols * sizeof(U));
  cudaMemcpy(
      d_ell_values, hA_values, m * A_ell_cols * sizeof(U), cudaMemcpyHostToDevice);
  // int nnz_per_row = nnz / m;
  // int total_warps = (m + warpSize - 1) / warpSize;
  // int threads_per_block=256;
  // int threads_per_row = prevPowerOf2(nnz_per_row);
  // int rows_per_block = threads_per_block >= threads_per_row
  //   ? threads_per_block / threads_per_row : 1;
  // int num_blocks = (m + rows_per_block - 1) / rows_per_block;  

  // int * drows = row_ptr;
  // int * dcols = (int *)coo->col_data;
  // U * dvals = (U *)coo->val_data;
  // uint32_t *dmax_nnz_per_row_per_warp;
  // int *drow_warp_mapping;
  // int *drow_in_warp;
  // cudaMalloc((void **)&dmax_nnz_per_row_per_warp,
  //              sizeof(uint32_t) * total_warps);
  // cudaMemset(dmax_nnz_per_row_per_warp, 0, sizeof(uint32_t) * total_warps);
  // cudaMalloc((void **)&drow_warp_mapping, sizeof(int) * m);
  // cudaMemset(drow_warp_mapping, 0, sizeof(int) * m);
  // cudaMalloc((void **)&drow_in_warp, sizeof(int) * m);
  // cudaMemset(drow_in_warp, 0, sizeof(int) * m);

  // int nb = (m + threads_per_block - 1) / threads_per_block;
  // computeMaxNnzPerRowPerBlock<<<dim3(nb, 1, 1),dim3(threads_per_block, 1, 1), 0, 0>>>
  //   (m, total_warps, drows, dmax_nnz_per_row_per_warp, drow_warp_mapping, drow_in_warp);

  // /* Step 2. reduce the max nnz_per_row_per_warp to get the total amount of
  //  * memory needed */
  // uint32_t *dmax_nnz_per_row_per_warp_scanned;
  // cudaMalloc((void **)&dmax_nnz_per_row_per_warp_scanned,
  //              sizeof(uint32_t) * (total_warps + 1));
  // cudaMemset(dmax_nnz_per_row_per_warp_scanned, 0,
  //              sizeof(uint32_t) * (total_warps + 1));
  // thrust::exclusive_scan(thrust::device, dmax_nnz_per_row_per_warp,
  //                 dmax_nnz_per_row_per_warp + total_warps + 1,
  //                 dmax_nnz_per_row_per_warp_scanned);
  // int total_nnz = 0;
  // cudaMemcpy(&total_nnz,
  //              dmax_nnz_per_row_per_warp_scanned + total_warps,
  //              sizeof(int), cudaMemcpyDeviceToHost);

  // /* Step 3. Fill the data structures */
  // double fill = (1.0 * total_nnz * warpSize - nnz) / (total_nnz * warpSize);
  // printf("total_nnz %d\n", total_nnz);
  // uint32_t *dellcols = NULL;
  // U *dellvals = NULL;
  // cudaMalloc((void **)&dellcols, sizeof(uint32_t) * total_nnz * warpSize);
  // cudaMalloc((void **)&dellvals, sizeof(U) * total_nnz * warpSize);
  // cudaMemset(dellcols, 0, sizeof(uint32_t) * total_nnz * warpSize);
  // cudaMemset(dellvals, 0, sizeof(U) * total_nnz * warpSize);
  // dim3 grid(num_blocks,1,1);
  // dim3 block(threads_per_block,1,1);
  // if (threads_per_row <= 2) {
  //   fillBlockColEll<2><<<grid,block>>>(m, total_warps,
  //                           dmax_nnz_per_row_per_warp_scanned, drow_warp_mapping,
  //                           drow_in_warp, drows, dcols, dvals, dellcols, dellvals);
  // } else if (threads_per_row <= 4) {
  //   fillBlockColEll<4><<<grid,block>>>(m, total_warps,
  //                           dmax_nnz_per_row_per_warp_scanned, drow_warp_mapping,
  //                           drow_in_warp, drows, dcols, dvals, dellcols, dellvals);
  // } else if (threads_per_row <= 8) {
  //   fillBlockColEll<8><<<grid,block>>>(m, total_warps,
  //                           dmax_nnz_per_row_per_warp_scanned, drow_warp_mapping,
  //                           drow_in_warp, drows, dcols, dvals, dellcols, dellvals);
  // } else if (threads_per_row <= 16) {
  //   fillBlockColEll<16><<<grid,block>>>(m, total_warps,
  //                           dmax_nnz_per_row_per_warp_scanned, drow_warp_mapping,
  //                           drow_in_warp, drows, dcols, dvals, dellcols, dellvals);
  // } else if (threads_per_row <= 32) {
  //   fillBlockColEll<32><<<grid,block>>>(m, total_warps,
  //                           dmax_nnz_per_row_per_warp_scanned, drow_warp_mapping,
  //                           drow_in_warp, drows, dcols, dvals, dellcols, dellvals);
  // } else {
  //   fillBlockColEll<64><<<grid,block>>>(m, total_warps,
  //                           dmax_nnz_per_row_per_warp_scanned, drow_warp_mapping,
  //                           drow_in_warp, drows, dcols, dvals, dellcols, dellvals);
  // }
  // cudaFree(dmax_nnz_per_row_per_warp);
  // cudaFree(drow_warp_mapping);
  // cudaFree(drow_in_warp);
  
  // int A_ell_cols = total_nnz / m;
  
  alphasparseCreateBlockedEll(&bell,
                              m,
                              n,
                              blocksize,
                              A_ell_cols,
                              d_ell_col_inx,
                              d_ell_values,
                              coo->row_type,
                              coo->idx_base,
                              coo->data_type);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
