#pragma once

#include "../types.h"
#include "thread.h"

void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *Y, ALPHA_INT ldY);
void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *Y, ALPHA_INT ldY);
void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *Y, ALPHA_INT ldY);
void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *Y, ALPHA_INT ldY);

void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *Y, ALPHA_INT ldY);
void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *Y, ALPHA_INT ldY);
void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *Y, ALPHA_INT ldY);
void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *Y, ALPHA_INT ldY);

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *Y, ALPHA_INT ldY);
void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *Y, ALPHA_INT ldY);
void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *Y, ALPHA_INT ldY);

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *Y, ALPHA_INT ldY);
void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *Y, ALPHA_INT ldY);
void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *Y, ALPHA_INT ldY);

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *Y, ALPHA_INT ldY);
void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *Y, ALPHA_INT ldY);
void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *Y, ALPHA_INT ldY);

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *Y, ALPHA_INT ldY);
void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *Y, ALPHA_INT ldY);
void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *Y, ALPHA_INT ldY);













