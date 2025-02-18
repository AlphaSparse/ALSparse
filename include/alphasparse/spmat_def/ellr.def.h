#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_ELLR spmat_ellr_s_t
#else
#define ALPHA_SPMAT_ELLR spmat_ellr_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_ELLR spmat_ellr_c_t
#else
#define ALPHA_SPMAT_ELLR spmat_ellr_z_t
#endif
#endif

typedef struct {
    float *values; // 列主存储非零元
    ALPHA_INT *indices;
    ALPHA_INT *rows_length; // 记录行长
    ALPHA_INT rows;
    ALPHA_INT cols;
    ALPHA_INT ld;

    float *d_values; // 列主存储非零元
    ALPHA_INT *d_indices;
    ALPHA_INT *d_rows_length;
} spmat_ellr_s_t;

typedef struct {
    double *values; // 列主存储非零元
    ALPHA_INT *indices;
    ALPHA_INT *rows_length;
    ALPHA_INT rows;
    ALPHA_INT cols;
    ALPHA_INT ld;

    double *d_values; // 列主存储非零元
    ALPHA_INT *d_indices;
    ALPHA_INT *d_rows_length;
} spmat_ellr_d_t;

typedef struct {
    ALPHA_Complex8 *values; // 列主存储非零元
    ALPHA_INT *indices;
    ALPHA_INT *rows_length;
    ALPHA_INT rows;
    ALPHA_INT cols;
    ALPHA_INT ld;

    ALPHA_Complex8 *d_values; // 列主存储非零元
    ALPHA_INT *d_indices;
    ALPHA_INT *d_rows_length;
} spmat_ellr_c_t;

typedef struct {
    ALPHA_Complex16 *values; // 列主存储非零元
    ALPHA_INT *indices;
    ALPHA_INT *rows_length;
    ALPHA_INT rows;
    ALPHA_INT cols;
    ALPHA_INT ld;

    ALPHA_Complex16 *d_values; // 列主存储非零元
    ALPHA_INT *d_indices;
    ALPHA_INT *d_rows_length;
} spmat_ellr_z_t;
