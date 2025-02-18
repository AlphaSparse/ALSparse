#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_COOAOS spmat_cooaos_s_t
#else
#define ALPHA_SPMAT_COOAOS spmat_cooaos_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_COOAOS spmat_cooaos_c_t
#else
#define ALPHA_SPMAT_COOAOS spmat_cooaos_z_t
#endif
#endif

typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  // bool ordered;

  float     *values;       //array of nnz elements containing the data (floating point).
  ALPHA_INT *indx;         //array of 2 * nnz elements containing alternating row and column indices (integer).

  ALPHA_INT *d_indx;
  float     *d_values;
} spmat_cooaos_s_t;

typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  // bool ordered;

  double    *values;       //array of nnz elements containing the data (floating point).
  ALPHA_INT *indx;         //array of 2 * nnz elements containing alternating row and column indices (integer).

  ALPHA_INT *d_indx;
  double    *d_values;
} spmat_cooaos_d_t;

typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  // bool ordered;

  ALPHA_Complex8 *values;  //array of nnz elements containing the data (floating point).
  ALPHA_INT *indx;         //array of 2 * nnz elements containing alternating row and column indices (integer).

  ALPHA_INT      *d_indx;
  ALPHA_Complex8 *d_values;
} spmat_cooaos_c_t;

typedef struct {
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  // bool ordered;

  ALPHA_Complex16 *values;       //array of nnz elements containing the data (floating point).
  ALPHA_INT       *indx;         //array of 2 * nnz elements containing alternating row and column indices (integer).

  ALPHA_INT       *d_indx;
  ALPHA_Complex16 *d_values;
} spmat_cooaos_z_t;