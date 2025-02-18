#include "alphasparse/kernel.h"

extern "C" void __spmv_ell_serial_host_sse_float(const float alpha, const float beta,
                                      const int lrs, const int lre, const int m,
                                      const int width, const int *indices,
                                      const float *values, const float *x,
                                      float *y);
void __spmv_ell_serial_host_plain_float(const float alpha, const float beta,
                                        const int lrs, const int lre,
                                        const int m, const int width,
                                        const int *indices, const float *values,
                                        const float *x, float *y);
void __spmv_ell_serial_host_u4_float(const float alpha, const float beta,
                                     const int lrs, const int lre, const int m,
                                     const int width, const int *indices,
                                     const float *values, const float *x,
                                     float *y);
// sell float
void __spmv_sell_c4_serial_host_fma128_float(
    const float alpha, const float beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const float *values,
    const float *x, float *y);
void __spmv_sell_c8_serial_host_fma128_float(
    const float alpha, const float beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const float *values,
    const float *x, float *y);
void __spmv_sell_c8_serial_host_fma256_float(
    const float alpha, const float beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const float *values,
    const float *x, float *y);
// sell double
void __spmv_sell_c4_serial_host_fma256_double(
    const double alpha, const double beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const double *values,
    const double *x, double *y);
void __spmv_sell_c8_serial_host_fma256_double(
    const double alpha, const double beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const double *values,
    const double *x, double *y);
void __spmv_sell_c4_serial_host_fma128_double(
    const double alpha, const double beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const double *values,
    const double *x, double *y);
void __spmv_sell_c8_serial_host_fma128_double(
    const double alpha, const double beta, const int num_chunks,
    const int *chunks_start, const int *col_indices, const double *values,
    const double *x, double *y);