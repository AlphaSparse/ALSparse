#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_SKY spmat_sky_s_t
#else
#define ALPHA_SPMAT_SKY spmat_sky_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_SKY spmat_sky_c_t
#else
#define ALPHA_SPMAT_SKY spmat_sky_z_t
#endif
#endif

typedef struct {
  ALPHA_INT *pointers;
  float *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_s_t;

typedef struct {
  ALPHA_INT *pointers;
  double *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_d_t;

typedef struct {
  ALPHA_INT *pointers;
  ALPHA_Complex8 *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_c_t;

typedef struct {
  ALPHA_INT *pointers;
  ALPHA_Complex16 *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_z_t;
