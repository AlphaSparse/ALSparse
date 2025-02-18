#pragma once

#include <math.h>
#include <assert.h>
#include <memory.h>
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"

float CalEpisilon_s();
double CalEpisilon_d();

double InfiniteNorm_d(const ALPHA_INT n, const double *xa, const double *xb);
double GeNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const double *val);
double TrSyNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const double *val, struct alpha_matrix_descr descr);
double DiNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const double *val, struct alpha_matrix_descr descr);

double DesNorm1_d(ALPHA_INT rows, ALPHA_INT cols, const double *val, ALPHA_INT ldv, alphasparse_layout_t layout);
double DesDiffNorm1_d(const ALPHA_INT rows, const ALPHA_INT cols, const double *xa, const ALPHA_INT lda, const double *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

float InfiniteNorm_s(const ALPHA_INT n, const float *xa, const float *xb);
float GeNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const float *val);
float TrSyNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const float *val, struct alpha_matrix_descr descr);
float DiNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const float *val, struct alpha_matrix_descr descr);

float DesNorm1_s(ALPHA_INT rows, ALPHA_INT cols, const float *val, ALPHA_INT ldv, alphasparse_layout_t layout);
float DesDiffNorm1_s(const ALPHA_INT rows, const ALPHA_INT cols, const float *xa, const ALPHA_INT lda, const float *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

float InfiniteNorm_c(const ALPHA_INT n, const ALPHA_Complex8 *xa, const ALPHA_Complex8 *xb);
float GeNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const ALPHA_Complex8 *val);
float TrSyNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex8 *val, struct alpha_matrix_descr descr);
float DiNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex8 *val, struct alpha_matrix_descr descr);

float DesNorm1_c(ALPHA_INT rows, ALPHA_INT cols, const ALPHA_Complex8 *val, ALPHA_INT ldv, alphasparse_layout_t layout);
float DesDiffNorm1_c(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_Complex8 *xa, const ALPHA_INT lda, const ALPHA_Complex8 *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

double InfiniteNorm_z(const ALPHA_INT n, const ALPHA_Complex16 *xa, const ALPHA_Complex16 *xb);
double GeNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const ALPHA_Complex16 *val);
double TrSyNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex16 *val, struct alpha_matrix_descr descr);
double DiNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex16 *val, struct alpha_matrix_descr descr);

double DesNorm1_z(ALPHA_INT rows, ALPHA_INT cols, const ALPHA_Complex16 *val, ALPHA_INT ldv, alphasparse_layout_t layout);
double DesDiffNorm1_z(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_Complex16 *xa, const ALPHA_INT lda, const ALPHA_Complex16 *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

#ifndef index2
#define index2(y, x, ldx) ((x) + (ldx) * (y))
#endif // !index2

#ifndef index3
#define index3(z, y, x, ldy, ldx) index2(index2(z, y, ldy), x, ldx)
#endif // !index3

#ifndef index4
#define index4(d, c, b, a, ldc, ldb, lda) index2(index2(index2(d, c, ldc), b, ldb), a, lda)
#endif // !index4

#ifndef index2_long
#define index2_long(y, x, ldx) ((uint64_t)(x) + (uint64_t)(ldx) * (y))
#endif // !index2

#ifndef index3_long
#define index3_long(z, y, x, ldy, ldx) index2_long(index2_long(z, y, ldy), x, ldx)
#endif // !index3

#ifndef index4_long
#define index4_long(d, c, b, a, ldc, ldb, lda) index2_long(index2_long(index2_long(d, c, ldc), b, ldb), a, lda)
#endif // !index4

#ifndef alpha_min
#define alpha_min(x, y) ((x) < (y) ? (x) : (y))
#endif // !alpha_min

#ifndef alpha_max
#define alpha_max(x, y) ((x) < (y) ? (y) : (x))
#endif // !alpha_max
