#include "alphasparse/kernel.h"
void __spmv_csr_serial_host_avx2_float(const float alpha, const float beta,
                                       const int num_rows, const int *Ap,
                                       const int *Aj, const float *Ax,
                                       const float *x, float *y);
extern "C" void __spmv_csr_serial_host_avx2_double(const double alpha, const double beta,
                                        const int num_rows, const int *Ap,
                                        const int *Aj, const double *Ax,
                                        const double *x, double *y);
extern "C" void __spmv_csr_serial_host_sse_float(const float alpha, const float beta,
                                      const int num_rows, const int *Ap,
                                      const int *Aj, const float *Ax,
                                      const float *x, float *y);
void csrmv_sload_shuffle_hadd_float_128(const float alpha, const float beta,
                                      const int num_rows, const int *Ap,
                                      const int *Aj, const float *Ax,
                                      const float *x, float *y);
void csrmv_vgather_hadd_float_128(const float alpha, const float beta,
                                      const int num_rows, const int *Ap,
                                      const int *Aj, const float *Ax,
                                      const float *x, float *y);
void __spmv_csr_serial_host_gather4_float(const float alpha, const float beta,
                                      const int num_rows, const int *Ap,
                                      const int *Aj, const float *Ax,
                                      const float *x, float *y);  
void __spmv_csr_serial_host_plain_float(const float alpha, const float beta,
                                        const int num_rows, const int *Ap,
                                        const int *Aj, const float *Ax,
                                        const float *x, float *y);
extern "C" void __spmv_csr_serial_host_sse_complex_float(
    const ALPHA_Complex8 alpha, const ALPHA_Complex8 beta, const int num_rows,
    const int *Ap, const int *Aj, const ALPHA_Complex8 *Ax,
    const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

extern "C" void __spmv_csr_serial_host_sse_complex_double(
    const ALPHA_Complex16 alpha, const ALPHA_Complex16 beta, const int num_rows,
    const int *Ap, const int *Aj, const ALPHA_Complex16 *Ax,
    const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
void __spmv_csr_serial_host_sse_complex_double_unrolling2(
    const ALPHA_Complex16 alpha, const ALPHA_Complex16 beta, const int num_rows,
    const int *Ap, const int *Aj, const ALPHA_Complex16 *Ax,
    const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
