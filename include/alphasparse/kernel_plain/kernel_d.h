#pragma once

alphasparseStatus_t axpy_d_plain(const ALPHA_INT nz,  const double a,  const double* x,  const ALPHA_INT* indx,  double* y);
alphasparseStatus_t gthr_d_plain(const ALPHA_INT nz,	const double* y, double* x, const ALPHA_INT* indx);
alphasparseStatus_t gthrz_d_plain(const ALPHA_INT nz, double* y, double* x, const ALPHA_INT* indx);
alphasparseStatus_t rot_d_plain(const ALPHA_INT nz, double* x, const ALPHA_INT* indx, double* y, const double c, const double s);
alphasparseStatus_t sctr_d_plain(const ALPHA_INT nz, const double* x, const ALPHA_INT* indx, double* y);
double doti_d_plain(const ALPHA_INT nz,  const double* x,  const ALPHA_INT* indx, const double* y);