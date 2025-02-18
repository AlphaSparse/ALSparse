#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_COO spmat_coo_s_t
#else
#define ALPHA_SPMAT_COO spmat_coo_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_COO spmat_coo_c_t
#else
#define ALPHA_SPMAT_COO spmat_coo_z_t
#endif
#endif

typedef struct {
  float *values;
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;

  ALPHA_INT *d_rows_indx;
  ALPHA_INT *d_cols_indx;
  float     *d_values;
} spmat_coo_s_t;

typedef struct {
  double *values;
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;

  ALPHA_INT *d_rows_indx;
  ALPHA_INT *d_cols_indx;
  double    *d_values;
} spmat_coo_d_t;

typedef struct {
  ALPHA_Complex8 *values;
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;
  
  ALPHA_INT      *d_rows_indx;
  ALPHA_INT      *d_cols_indx;
  ALPHA_Complex8 *d_values;
} spmat_coo_c_t;

typedef struct {
  ALPHA_Complex16 *values;
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;
  
  ALPHA_INT       *d_rows_indx;
  ALPHA_INT       *d_cols_indx;
  ALPHA_Complex16 *d_values;
} spmat_coo_z_t;

template <typename TYPE>
alphasparseStatus_t coo_order(internal_spmat mat);
