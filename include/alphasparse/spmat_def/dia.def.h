#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_DIA spmat_dia_s_t
#else
#define ALPHA_SPMAT_DIA spmat_dia_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_DIA spmat_dia_c_t
#else
#define ALPHA_SPMAT_DIA spmat_dia_z_t
#endif
#endif

typedef struct {
  float *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;

  float *d_values;
  ALPHA_INT *d_distance;
} spmat_dia_s_t;

typedef struct {
  double *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;

  double *d_values;
  ALPHA_INT *d_distance;
} spmat_dia_d_t;

typedef struct {
  ALPHA_Complex8 *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;

  ALPHA_Complex8 *d_values;
  ALPHA_INT *d_distance;
} spmat_dia_c_t;

typedef struct {
  ALPHA_Complex16 *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;

  ALPHA_Complex16 *d_values;
  ALPHA_INT *d_distance;
} spmat_dia_z_t;
