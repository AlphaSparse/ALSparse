#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_BSR spmat_bsr_s_t
#else
#define ALPHA_SPMAT_BSR spmat_bsr_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_BSR spmat_bsr_c_t
#else
#define ALPHA_SPMAT_BSR spmat_bsr_z_t
#endif
#endif

typedef struct {
  float *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;

  float     *d_values;
  ALPHA_INT *d_rows_ptr;
  ALPHA_INT *d_col_indx;
} spmat_bsr_s_t;

typedef struct {
  double *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;
  
  double    *d_values;
  ALPHA_INT *d_rows_ptr;
  ALPHA_INT *d_col_indx;
} spmat_bsr_d_t;

typedef struct {
  ALPHA_Complex8 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;
  
  ALPHA_Complex8 *d_values;
  ALPHA_INT      *d_rows_ptr;
  ALPHA_INT      *d_col_indx;
} spmat_bsr_c_t;

typedef struct {
  ALPHA_Complex16 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;

  
  ALPHA_Complex16 *d_values;
  ALPHA_INT       *d_rows_ptr;
  ALPHA_INT       *d_col_indx;
} spmat_bsr_z_t;
