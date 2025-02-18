#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_CSR spmat_csr_s_t
#else
#define ALPHA_SPMAT_CSR spmat_csr_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_CSR spmat_csr_c_t
#else
#define ALPHA_SPMAT_CSR spmat_csr_z_t
#endif
#endif

typedef struct {
  float *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  float     *d_values;
  ALPHA_INT *d_row_ptr;
  ALPHA_INT *d_col_indx;

} spmat_csr_s_t;

typedef struct {
  double *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  double    *d_values;
  ALPHA_INT *d_row_ptr;
  ALPHA_INT *d_col_indx;
} spmat_csr_d_t;

typedef struct {
  ALPHA_Complex8 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  ALPHA_Complex8 *d_values;
  ALPHA_INT      *d_row_ptr;
  ALPHA_INT      *d_col_indx;
} spmat_csr_c_t;

typedef struct {
  ALPHA_Complex16 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  ALPHA_Complex16 *d_values;
  ALPHA_INT       *d_row_ptr;
  ALPHA_INT       *d_col_indx;
} spmat_csr_z_t;
