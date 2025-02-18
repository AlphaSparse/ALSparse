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
#define ALPHA_SPMAT_COO spmat_coo_s_t
#else
#define ALPHA_SPMAT_COO spmat_coo_h_t
#endif
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

template <typename T>
struct spmat_coo_t
{
    T *values;
    int *row_indx;
    int *col_indx;
    int rows;
    int cols;
    int nnz;
    bool ordered;

    int *d_rows_indx;
    int *d_cols_indx;
    T     *d_values;
};

typedef struct {
  float *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  int *d_rows_indx;
  int *d_cols_indx;
  float     *d_values;
} spmat_coo_s_t;

typedef struct {
  double *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  int *d_rows_indx;
  int *d_cols_indx;
  double    *d_values;
} spmat_coo_d_t;

#ifdef __CUDA__
typedef struct {
  half *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;

  int *d_rows_indx;
  int *d_cols_indx;
  half     *d_values;
} spmat_coo_h_t;

typedef struct {
  cuFloatComplex *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;
  
  int      *d_rows_indx;
  int      *d_cols_indx;
  cuFloatComplex *d_values;
} spmat_coo_c_t;

typedef struct {
  cuDoubleComplex *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;
  
  int       *d_rows_indx;
  int       *d_cols_indx;
  cuDoubleComplex *d_values;
} spmat_coo_z_t;
#endif

#ifdef __HIP__
typedef struct {
  hipFloatComplex *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;
  
  int      *d_rows_indx;
  int      *d_cols_indx;
  hipFloatComplex *d_values;
} spmat_coo_c_t;

typedef struct {
  hipDoubleComplex *values;
  int *row_indx;
  int *col_indx;
  int rows;
  int cols;
  int nnz;
  bool ordered;
  
  int       *d_rows_indx;
  int       *d_cols_indx;
  hipDoubleComplex *d_values;
} spmat_coo_z_t;
#endif