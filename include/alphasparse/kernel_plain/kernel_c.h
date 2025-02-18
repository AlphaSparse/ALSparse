#pragma once

alphasparseStatus_t axpy_c_plain(const ALPHA_INT nz,  const ALPHA_Complex8 a,  const ALPHA_Complex8* x,  const ALPHA_INT* indx,  ALPHA_Complex8* y);
alphasparseStatus_t gthr_c_plain(const ALPHA_INT nz,	const ALPHA_Complex8* y, ALPHA_Complex8* x, const ALPHA_INT* indx);
alphasparseStatus_t gthrz_c_plain(const ALPHA_INT nz, ALPHA_Complex8* y, ALPHA_Complex8* x, const ALPHA_INT* indx);
alphasparseStatus_t sctr_c_plain(const ALPHA_INT nz, const ALPHA_Complex8* x, const ALPHA_INT* indx, ALPHA_Complex8* y);
void dotci_c_sub_plain(const ALPHA_INT nz,  const ALPHA_Complex8* x,  const ALPHA_INT* indx, const ALPHA_Complex8* y, ALPHA_Complex8 *dutci);
void dotui_c_sub_plain(const ALPHA_INT nz,  const ALPHA_Complex8* x,  const ALPHA_INT* indx, const ALPHA_Complex8* y, ALPHA_Complex8 *dutui);
