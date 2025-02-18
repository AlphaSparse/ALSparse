#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_CSR5 spmat_csr5_s_t
#else
#define ALPHA_SPMAT_CSR5 spmat_csr5_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_CSR5 spmat_csr5_c_t
#else
#define ALPHA_SPMAT_CSR5 spmat_csr5_z_t
#endif
#endif

#define ALPHA_CSR5_OMEGA 32

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  float     *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  float     *calibrator;              // opt: CSR5 calibrator CPU case
  
  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  float     *d_val;

  uint32_t  *d_tile_ptr;
  uint32_t  *d_tile_desc;
  ALPHA_INT *d_tile_desc_offset_ptr;
  ALPHA_INT *d_tile_desc_offset;
  float     *d_calibrator;

} spmat_csr5_s_t;

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  double    *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  double    *calibrator;              // opt: CSR5 calibrator CPU case
  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  double     *d_val;

  uint32_t  *d_tile_ptr;
  uint32_t  *d_tile_desc;
  ALPHA_INT *d_tile_desc_offset_ptr;
  ALPHA_INT *d_tile_desc_offset;
  double    *d_calibrator;
} spmat_csr5_d_t;

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  ALPHA_Complex8     *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  ALPHA_Complex8     *calibrator;              // opt: CSR5 calibrator CPU case
  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  ALPHA_Complex8     *d_val;

  uint32_t       *d_tile_ptr;
  uint32_t       *d_tile_desc;
  ALPHA_INT      *d_tile_desc_offset_ptr;
  ALPHA_INT      *d_tile_desc_offset;
  ALPHA_Complex8 *d_calibrator;
} spmat_csr5_c_t;

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  ALPHA_Complex16     *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  ALPHA_Complex16    *calibrator;              // opt: CSR5 calibrator CPU case
  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  ALPHA_Complex16     *d_val;

  uint32_t        *d_tile_ptr;
  uint32_t        *d_tile_desc;
  ALPHA_INT       *d_tile_desc_offset_ptr;
  ALPHA_INT       *d_tile_desc_offset;
  ALPHA_Complex16 *d_calibrator;
} spmat_csr5_z_t;