// #include "alphasparse/types.h"
#ifdef __CUDA__
#include <cuComplex.h>
#include <cuda_fp16.h>
#endif
#ifdef __HIP__
#include <hip/hip_complex.h>
#endif
#ifndef COMPLEX
#ifndef DOUBLE
#ifndef HALF
#define ALPHA_SPMAT_CSR spmat_csr_s_t
#else
#define ALPHA_SPMAT_CSR spmat_csr_h_t
#endif
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

template <typename T>
struct spmat_csr_t{
  T *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  bool ordered;

  T     *d_values;
  int *d_row_ptr;
  int *d_col_indx;
};

typedef struct {
  float *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  float     *d_values;
  int *d_row_ptr;
  int *d_col_indx;

} spmat_csr_s_t;

typedef struct {
  double *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  double    *d_values;
  int *d_row_ptr;
  int *d_col_indx;
} spmat_csr_d_t;

#ifdef __CUDA__
typedef struct {
  half *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  bool ordered;

  half     *d_values;
  int *d_row_ptr;
  int *d_col_indx;

} spmat_csr_h_t;

typedef struct {
  cuFloatComplex *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  cuFloatComplex *d_values;
  int      *d_row_ptr;
  int      *d_col_indx;
} spmat_csr_c_t;

typedef struct {
  cuDoubleComplex *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  cuDoubleComplex *d_values;
  int       *d_row_ptr;
  int       *d_col_indx;
} spmat_csr_z_t;
#endif

#ifdef __HIP__
typedef struct {
  hipFloatComplex *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  hipFloatComplex *d_values;
  int      *d_row_ptr;
  int      *d_col_indx;
} spmat_csr_c_t;

typedef struct {
  hipDoubleComplex *values;
  int *rows_start;
  int *rows_end;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  hipDoubleComplex *d_values;
  int       *d_row_ptr;
  int       *d_col_indx;
} spmat_csr_z_t;
#endif