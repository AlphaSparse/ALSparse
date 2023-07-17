#pragma once
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"

int check(const int32_t *answer_data, size_t answer_size, const int32_t *result_data,
          size_t result_size);
int check(const int8_t *answer_data, size_t answer_size, const int8_t *result_data,
          size_t result_size);
int check(const half *answer_data, size_t answer_size, const half *result_data,
          size_t result_size);
int check(const half2 *answer_data, size_t answer_size, const half2 *result_data, size_t result_size);
int check(const nv_bfloat162 *answer_data, size_t answer_size, const nv_bfloat162 *result_data,
          size_t result_size);
int check(const nv_bfloat16 *answer_data, size_t answer_size, const nv_bfloat16 *result_data,
          size_t result_size);
int check(const float *answer_data, size_t answer_size, const float *result_data, size_t result_size);
int check(const double *answer_data, size_t answer_size, const double *result_data, size_t result_size);
int check(const cuFloatComplex *answer_data, size_t answer_size, const cuFloatComplex *result_data, size_t result_size);
int check(const cuDoubleComplex *answer_data, size_t answer_size, const cuDoubleComplex *result_data, size_t result_size);

int check_s_l2(const float *answer_data, size_t answer_size, const float *result_data, size_t result_size, const float *x, const float *y, const float alpha, const float beta, int argc, const char *argv[]);
int check_s_l3(const float *answer_data, const int ldans, size_t answer_size, const float *result_data, const int ldres, size_t result_size, const int *res_col_indx, const float *x, const int ldx, const float *y, const int ldy, const float alpha, const float beta, int argc, const char *argv[]);
int check_d_l2(const double *answer_data, size_t answer_size, const double *result_data, size_t result_size, const double *x, const double *y, const double alpha, const double beta, int argc, const char *argv[]);
int check_d_l3(const double *answer_data, const int ldans, size_t answer_size, const double *result_data, const int ldres, size_t result_size, const int *res_col_indx, const double *x, const int ldx, const double *y, const int ldy, const double alpha, const double beta, int argc, const char *argv[]);
int check_c_l2(const cuFloatComplex *answer_data, size_t answer_size, const cuFloatComplex *result_data, size_t result_size, const cuFloatComplex *x, const cuFloatComplex *y, const cuFloatComplex alpha, const cuFloatComplex beta, int argc, const char *argv[]);
int check_c_l3(const cuFloatComplex *answer_data, const int ldans, size_t answer_size, const cuFloatComplex *result_data, const int ldres, size_t result_size, const int *res_col_indx, const cuFloatComplex *x, const int ldx, const cuFloatComplex *y, const int ldy, const cuFloatComplex alpha, const cuFloatComplex beta, int argc, const char *argv[]);
int check_z_l2(const cuDoubleComplex *answer_data, size_t answer_size, const cuDoubleComplex *result_data, size_t result_size, const cuDoubleComplex *x, const cuDoubleComplex *y, const cuDoubleComplex alpha, const cuDoubleComplex beta, int argc, const char *argv[]);
int check_z_l3(const cuDoubleComplex *answer_data, const int ldans, size_t answer_size, const cuDoubleComplex *result_data, const int ldres, size_t result_size, const int *res_col_indx, const cuDoubleComplex *x, const int ldx, const cuDoubleComplex *y, const int ldy, const cuDoubleComplex alpha, const cuDoubleComplex beta, int argc, const char *argv[]);

void check_int_vec(int *answer_data, int size_ans, int *result_data, int size_res);