#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_SELL_C_SIGMA spmat_sell_csigma_s_t
#else
#define ALPHA_SPMAT_SELL_C_SIGMA spmat_sell_csigma_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_SELL_C_SIGMA spmat_sell_csigma_c_t
#else
#define ALPHA_SPMAT_SELL_C_SIGMA spmat_sell_csigma_z_t
#endif
#endif

#define NUM_BINS 5

/*

assume SIGMA=2, C=2
  original matrix =>
         0, 1, 2, 3
    0: [ 1, 0, 1, 0,
    1:   1, 0, 0, 0,
    2:   1, 0, 1, 1,
    3:   1, 1, 1, 1 ]

  1. sort by row length every SIGMA rows =>

         0, 1, 2, 3
    1: [ 1, 0, 0, 0,
    0:   1, 0, 1, 0,
    2:   1, 0, 1, 1,
    3:   1, 1, 1, 1 ]

  2. compress and uniform row length within one chunk =>

    1: [ 1, 0,               1: [ 0, -1,
    0:   1, 1,               0:   0, 2,
    2:   1, 1, 1, 0          2:   0, 2, 3, -1,
    3:   1, 1, 1, 1 ]        3:   0, 1, 2, 3 ]
         values                   indices

  3. split up (col major)   =>

    1: [ 1,0,                1: [ 0, -1,
    0:   1,1 ]               0:   0, 2 ]

    2: [ 1,1,1,0,            2: [ 0, 2, 3, -1,
    3:   1,1,1,1 ]           3:   0, 1, 2, 3 ]
      values                      indices            row_offset => [ 0, 4, 12 ]
*/
// row length <= 8 are so-called special rows
// ( those special rows does not constitute the sell_c_sigma)
// and those rows are categorized into 5 bins: [0],[1],[2],[3,4],[5,8] )
typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT rows_sell_end;
  ALPHA_INT rows_len_break;

  ALPHA_INT num_special_bins;  // =NUM_BINS (5)
  double *bins_padding_ratio;  //

  // bin part
  ALPHA_INT *bins_row_start;
  ALPHA_INT *bins_row_end;
  ALPHA_INT *bins_nz_start;   // start offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_nz_end;     // end offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_indices;
  float *bins_values;

  ALPHA_INT *rows_indx;  // record orignal row index

  // sell part
  ALPHA_INT C;           // for those length > 8
  ALPHA_INT SIGMA;       // for those length > 8
  ALPHA_INT num_chunks;  // for those length > 8
  ALPHA_INT *indices;
  float *values;           // nz values stored chunk by chunk
  ALPHA_INT *rows_length;  // width of each chunk (dont use by now)

  ALPHA_INT *chunks_start;    // start offset of each chunk : dim num_chunk
  ALPHA_INT *chunks_end;      // end offset of each chunk : dim num_chunk
  double sell_padding_ratio;  // stored nnz / actual nnz

  double total_padding_ratio;  // stored nnz / actual nnz

  float *d_values;           // nz values stored chunk by chunk
  ALPHA_INT *d_rows_indx;    // start offset of each chunk
  ALPHA_INT *d_rows_length;  // width of each chunk
  ALPHA_INT *d_indices;
  ALPHA_INT *d_chunks_start;  // start offset of current chunk
  ALPHA_INT *d_chunks_end;    // end offset of current chunk
} spmat_sell_csigma_s_t;
typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT rows_sell_end;
  ALPHA_INT rows_len_break;

  ALPHA_INT num_special_bins;  // =NUM_BINS (5)
  double *bins_padding_ratio;  //

  // bin part
  ALPHA_INT *bins_row_start;
  ALPHA_INT *bins_row_end;
  ALPHA_INT *bins_nz_start;   // start offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_nz_end;     // end offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_indices;
  double *bins_values;

  ALPHA_INT *rows_indx;  // record orignal row index

  // sell part
  ALPHA_INT C;           // for those length > 8
  ALPHA_INT SIGMA;       // for those length > 8
  ALPHA_INT num_chunks;  // for those length > 8
  ALPHA_INT *indices;
  double *values;           // nz values stored chunk by chunk
  ALPHA_INT *rows_length;  // width of each chunk (dont use by now)

  ALPHA_INT *chunks_start;    // start offset of each chunk : dim num_chunk
  ALPHA_INT *chunks_end;      // end offset of each chunk : dim num_chunk
  double sell_padding_ratio;  // stored nnz / actual nnz

  double total_padding_ratio;  // stored nnz / actual nnz

  double *d_values;           // nz values stored chunk by chunk
  ALPHA_INT *d_rows_indx;    // start offset of each chunk
  ALPHA_INT *d_rows_length;  // width of each chunk
  ALPHA_INT *d_indices;
  ALPHA_INT *d_chunks_start;  // start offset of current chunk
  ALPHA_INT *d_chunks_end;    // end offset of current chunk
} spmat_sell_csigma_d_t;

typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT rows_sell_end;
  ALPHA_INT rows_len_break;

  ALPHA_INT num_special_bins;  // =NUM_BINS (5)
  double *bins_padding_ratio;  //

  // bin part
  ALPHA_INT *bins_row_start;
  ALPHA_INT *bins_row_end;
  ALPHA_INT *bins_nz_start;   // start offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_nz_end;     // end offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_indices;
  ALPHA_Complex8 *bins_values;

  ALPHA_INT *rows_indx;  // record orignal row index

  // sell part
  ALPHA_INT C;           // for those length > 8
  ALPHA_INT SIGMA;       // for those length > 8
  ALPHA_INT num_chunks;  // for those length > 8
  ALPHA_INT *indices;
  ALPHA_Complex8 *values;           // nz values stored chunk by chunk
  ALPHA_INT *rows_length;  // width of each chunk (dont use by now)

  ALPHA_INT *chunks_start;    // start offset of each chunk : dim num_chunk
  ALPHA_INT *chunks_end;      // end offset of each chunk : dim num_chunk
  double sell_padding_ratio;  // stored nnz / actual nnz

  double total_padding_ratio;  // stored nnz / actual nnz

  ALPHA_Complex8 *d_values;           // nz values stored chunk by chunk
  ALPHA_INT *d_rows_indx;    // start offset of each chunk
  ALPHA_INT *d_rows_length;  // width of each chunk
  ALPHA_INT *d_indices;
  ALPHA_INT *d_chunks_start;  // start offset of current chunk
  ALPHA_INT *d_chunks_end;    // end offset of current chunk
} spmat_sell_csigma_c_t;

typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT rows_sell_end;
  ALPHA_INT rows_len_break;

  ALPHA_INT num_special_bins;  // =NUM_BINS (5)
  double *bins_padding_ratio;  //

  // bin part
  ALPHA_INT *bins_row_start;
  ALPHA_INT *bins_row_end;
  ALPHA_INT *bins_nz_start;   // start offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_nz_end;     // end offset of each bins : dim num_singular_bins
  ALPHA_INT *bins_indices;
  ALPHA_Complex16 *bins_values;

  ALPHA_INT *rows_indx;  // record orignal row index

  // sell part
  ALPHA_INT C;           // for those length > 8
  ALPHA_INT SIGMA;       // for those length > 8
  ALPHA_INT num_chunks;  // for those length > 8
  ALPHA_INT *indices;
  ALPHA_Complex16 *values;           // nz values stored chunk by chunk
  ALPHA_INT *rows_length;  // width of each chunk (dont use by now)

  ALPHA_INT *chunks_start;    // start offset of each chunk : dim num_chunk
  ALPHA_INT *chunks_end;      // end offset of each chunk : dim num_chunk
  double sell_padding_ratio;  // stored nnz / actual nnz

  double total_padding_ratio;  // stored nnz / actual nnz

  ALPHA_Complex16 *d_values;           // nz values stored chunk by chunk
  ALPHA_INT *d_rows_indx;    // start offset of each chunk
  ALPHA_INT *d_rows_length;  // width of each chunk
  ALPHA_INT *d_indices;
  ALPHA_INT *d_chunks_start;  // start offset of current chunk
  ALPHA_INT *d_chunks_end;    // end offset of current chunk
} spmat_sell_csigma_z_t;
