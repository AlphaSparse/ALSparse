#include "./ellmv/ellmv_kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <string.h>

// bin length : [8],[4],[2],[1],[0]
inline static void bins1_mv(const TYPE alpha, const ALPHA_INT rs,
                            const ALPHA_INT *row_indx, const ALPHA_INT re,
                            const ALPHA_INT *bin_indices,
                            TYPE *bin_values, const TYPE *x,
                            TYPE *y, const TYPE beta) {
#ifdef S
  const ALPHA_INT lre16 = (re >> 4) << 4;
  const ALPHA_INT lre8 = (re >> 3) << 3;
  const ALPHA_INT lre4 = (re >> 2) << 2;
  ALPHA_INT ldbin = 4;
  __m128 v_alpha, v_beta;
  const TYPE *val = bin_values;
  const ALPHA_INT *indices = bin_indices;
  v_alpha = _mm_set1_ps(alpha);
  v_beta = _mm_set1_ps(beta);
  __m128 v_val0, v_val1, v_val2, v_val3;
  __m128 v_x0, v_x1, v_x2, v_x3;
  __m128 x0, x1, x2, x3, x4, x5, x6, x7;
  __m128 tmp0, tmp1, tmp2, tmp3;
  __m128 tmp4, tmp5, tmp6, tmp7;
  __m128 v_accum;
  __m128 v_accum0;
  __m128 v_accum1;
  int idx0, idx1, idx2, idx3;
  int idx4, idx5, idx6, idx7;
  __m128 v_y0, v_y1, v_y2, v_y3;
  ALPHA_INT r = 0;
  // chunksize = 8, 4
  for (; r < lre8; r += 8) {
    v_val0 = _mm_loadu_ps(val + 0);
    v_val1 = _mm_loadu_ps(val + 4);

    // v_x0, v_x1
    x0 = _mm_load_ss(x + indices[0]);
    x1 = _mm_load_ss(x + indices[1]);
    x2 = _mm_load_ss(x + indices[2]);
    x3 = _mm_load_ss(x + indices[3]);
    x4 = _mm_load_ss(x + indices[4]);
    x5 = _mm_load_ss(x + indices[5]);
    x6 = _mm_load_ss(x + indices[6]);
    x7 = _mm_load_ss(x + indices[7]);

    tmp0 = _mm_shuffle_ps(x0, x1, 0);
    tmp1 = _mm_shuffle_ps(x2, x3, 0);
    v_x0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    tmp2 = _mm_shuffle_ps(x4, x5, 0);
    tmp3 = _mm_shuffle_ps(x6, x7, 0);
    v_x1 = _mm_shuffle_ps(tmp2, tmp3, 0x88);

    v_accum0 = _mm_mul_ps(v_x0, v_val0);
    v_accum1 = _mm_mul_ps(v_x1, v_val1);

    v_y0 = _mm_loadu_ps(y + r);
    v_y1 = _mm_loadu_ps(y + r + 4);

    v_y0 = _mm_mul_ps(v_y0, v_beta);
    v_y1 = _mm_mul_ps(v_y1, v_beta);

    v_y0 = _mm_fmadd_ps(v_accum0, v_alpha, v_y0);
    v_y1 = _mm_fmadd_ps(v_accum1, v_alpha, v_y1);

    _mm_storeu_ps(y + r, v_y0);
    _mm_storeu_ps(y + r + 4, v_y1);

    val += 8;
    indices += 8;
  }

  for (; r < lre4; r += 4) {
    v_val0 = _mm_loadu_ps(val + 0);
    // v_x0, v_x1
    x0 = _mm_load_ss(x + indices[0]);
    x1 = _mm_load_ss(x + indices[1]);
    x2 = _mm_load_ss(x + indices[2]);
    x3 = _mm_load_ss(x + indices[3]);

    tmp0 = _mm_shuffle_ps(x0, x1, 0);
    tmp1 = _mm_shuffle_ps(x2, x3, 0);
    v_x0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    v_accum0 = _mm_mul_ps(v_x0, v_val0);

    v_y0 = _mm_loadu_ps(y + r);
    v_y0 = _mm_mul_ps(v_y0, v_beta);

    v_y0 = _mm_fmadd_ps(v_accum0, v_alpha, v_y0);
    _mm_storeu_ps(y + r, v_y0);

    val += 4;
    indices += 4;
  }
#else
  for (ALPHA_INT r = 0; r < re; r++) {
    TYPE tmp;
    alpha_setzero(tmp);
    alpha_madde(tmp, bin_values[0], x[bin_indices[0]]);
    y[r] = alpha_mul(y[r], beta);
    alpha_madde(y[r], tmp, alpha);
    bin_values += 1;
    bin_indices += 1;
  }
#endif
}

inline static void bins2_mv(const TYPE alpha, const ALPHA_INT rs,
                            const ALPHA_INT *row_indx, const ALPHA_INT re,
                            const ALPHA_INT *bin_indices,
                            TYPE *bin_values, const TYPE *x,
                            TYPE *y, const TYPE beta) {
#ifdef S
  ALPHA_INT lre4 = (re >> 2) << 2;
  ALPHA_INT lre8 = (re >> 3) << 3;
  __m128 v_alpha, v_beta;
  const TYPE *val = bin_values;
  const ALPHA_INT *indices = bin_indices;
  v_alpha = _mm_set1_ps(alpha);
  v_beta = _mm_set1_ps(beta);
  // memset(bin_indices,0,sizeof(ALPHA_INT) * re * 8);
  __m128 v_val0, v_val1, v_val2, v_val3;
  __m128 v_x0, v_x1, v_x2, v_x3;
  __m128 x0, x1, x2, x3, x4, x5, x6, x7;
  __m128 tmp0, tmp1, tmp2, tmp3;
  __m128 tmp4, tmp5, tmp6, tmp7;
  __m128 v_accum;
  __m128 v_accum0,v_accum1,v_accum2,v_accum3;
  int idx0, idx1, idx2, idx3;
  int idx4, idx5, idx6, idx7;
  __m128 v_y0,v_y1;
  ALPHA_INT r = 0;
  // chunksize = 8,4
  for (; r < lre8; r += 8) {

    v_val0 = _mm_loadu_ps(val + 0);
    v_val1 = _mm_loadu_ps(val + 4);
    v_val2 = _mm_loadu_ps(val + 8);
    v_val3 = _mm_loadu_ps(val + 12);
    
    tmp0 = _mm_shuffle_ps(_mm_load_ss(x + indices[0]), _mm_load_ss(x + indices[1]), 0);
    tmp1 = _mm_shuffle_ps(_mm_load_ss(x + indices[2]), _mm_load_ss(x + indices[3]), 0);
    v_x0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    tmp2 = _mm_shuffle_ps(_mm_load_ss(x + indices[4]), _mm_load_ss(x + indices[5]), 0);
    tmp3 = _mm_shuffle_ps(_mm_load_ss(x + indices[6]), _mm_load_ss(x + indices[7]), 0);
    v_x1 = _mm_shuffle_ps(tmp2, tmp3, 0x88);

    v_accum0 = _mm_mul_ps(v_x0, v_val0);
    v_accum1 = _mm_mul_ps(v_x1, v_val1);

    tmp4 = _mm_shuffle_ps(_mm_load_ss(x + indices[8]), _mm_load_ss(x + indices[9]), 0);
    tmp5 = _mm_shuffle_ps(_mm_load_ss(x + indices[10]), _mm_load_ss(x + indices[11]), 0);
    v_x2 = _mm_shuffle_ps(tmp4, tmp5, 0x88);

    tmp6 = _mm_shuffle_ps(_mm_load_ss(x + indices[12]), _mm_load_ss(x + indices[13]), 0);
    tmp7 = _mm_shuffle_ps(_mm_load_ss(x + indices[14]), _mm_load_ss(x + indices[15]), 0);
    v_x3 = _mm_shuffle_ps(tmp6, tmp7, 0x88);

    v_accum2 = _mm_mul_ps(v_x2, v_val2);
    v_accum3 = _mm_mul_ps(v_x3, v_val3);

    v_accum0 = _mm_add_ps(v_accum0, v_accum1);
    v_accum2 = _mm_add_ps(v_accum2, v_accum3);

    v_y0 = _mm_loadu_ps(y + r);
    v_y1 = _mm_loadu_ps(y + r + 4);
    v_y0 = _mm_mul_ps(v_y0, v_beta);
    v_y1 = _mm_mul_ps(v_y1, v_beta);

    v_y0 = _mm_fmadd_ps(v_alpha,v_accum0,v_y0);
    v_y1 = _mm_fmadd_ps(v_alpha,v_accum2,v_y1);

    _mm_storeu_ps(y + r, v_y0);
    _mm_storeu_ps(y + r + 4, v_y1);

    val += 16;
    indices += 16;
  }
  for (; r < lre4; r += 4) {
    v_accum0 = _mm_setzero_ps();
    v_accum1 = _mm_setzero_ps();

    v_val0 = _mm_loadu_ps(val + 0);
    v_val1 = _mm_loadu_ps(val + 4);

    // v_x0, v_x1
    x0 = _mm_load_ss(x + indices[0]);
    x1 = _mm_load_ss(x + indices[1]);
    x2 = _mm_load_ss(x + indices[2]);
    x3 = _mm_load_ss(x + indices[3]);
    x4 = _mm_load_ss(x + indices[4]);
    x5 = _mm_load_ss(x + indices[5]);
    x6 = _mm_load_ss(x + indices[6]);
    x7 = _mm_load_ss(x + indices[7]);

    tmp0 = _mm_shuffle_ps(x0, x1, 0);
    tmp1 = _mm_shuffle_ps(x2, x3, 0);
    v_x0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    tmp2 = _mm_shuffle_ps(x4, x5, 0);
    tmp3 = _mm_shuffle_ps(x6, x7, 0);
    v_x1 = _mm_shuffle_ps(tmp2, tmp3, 0x88);

    v_accum0 = _mm_mul_ps(v_x0, v_val0);
    v_accum1 = _mm_mul_ps(v_x1, v_val1);

    v_y0 = _mm_loadu_ps(y + r);
    v_y0 = _mm_mul_ps(v_y0, v_beta);
    v_accum = _mm_add_ps(v_accum0, v_accum1);
    v_y0 = _mm_fmadd_ps(v_alpha,v_accum,v_y0);
    _mm_storeu_ps(y + r, v_y0);

    val += 8;
    indices += 8;
  }
#else
  for (ALPHA_INT r = 0; r < re; r++) {
    TYPE tmp;
    alpha_setzero(tmp);
    alpha_madde(tmp, bin_values[0], x[bin_indices[0]]);
    alpha_madde(tmp, bin_values[1], x[bin_indices[1]]);
    y[r] = alpha_mul(y[r], beta);
    alpha_madde(y[r], tmp, alpha);
    bin_values += 2;
    bin_indices += 2;
  }
#endif
}

inline static void bins4_mv(const TYPE alpha, const ALPHA_INT rs,
                            const ALPHA_INT *row_indx, const ALPHA_INT re,
                            const ALPHA_INT *bin_indices,
                            TYPE *bin_values, const TYPE *x,
                            TYPE *y, const TYPE beta) {
#ifdef S
  ALPHA_INT lre = (re >> 2) << 2;
  ALPHA_INT ldbin = 4;
  __m128 v_alpha, v_beta;
  const TYPE *val = bin_values;
  const ALPHA_INT *indices = bin_indices;
  v_alpha = _mm_set1_ps(alpha);
  v_beta = _mm_set1_ps(beta);
  // memset(bin_indices,0,sizeof(ALPHA_INT) * re * 8);
  __m128 v_val0, v_val1, v_val2, v_val3;
  __m128 v_x0, v_x1, v_x2, v_x3;
  __m128 x0, x1, x2, x3, x4, x5, x6, x7;
  __m128 tmp0, tmp1, tmp2, tmp3;
  __m128 tmp4, tmp5, tmp6, tmp7;
  __m128 v_accum;
  __m128 v_accum0;
  __m128 v_accum1;
  int idx0, idx1, idx2, idx3;
  int idx4, idx5, idx6, idx7;
  __m128 v_y;
  // chunksize = 4
  for (ALPHA_INT r = 0; r < lre; r += 4) {
    v_accum0 = _mm_setzero_ps();
    v_accum1 = _mm_setzero_ps();

    v_val0 = _mm_loadu_ps(val + 0);
    v_val1 = _mm_loadu_ps(val + 4);
    v_val2 = _mm_loadu_ps(val + 8);
    v_val3 = _mm_loadu_ps(val + 12);

    // v_x0, v_x1
    x0 = _mm_load_ss(x + indices[0]);
    x1 = _mm_load_ss(x + indices[1]);
    x2 = _mm_load_ss(x + indices[2]);
    x3 = _mm_load_ss(x + indices[3]);
    x4 = _mm_load_ss(x + indices[4]);
    x5 = _mm_load_ss(x + indices[5]);
    x6 = _mm_load_ss(x + indices[6]);
    x7 = _mm_load_ss(x + indices[7]);

    tmp0 = _mm_shuffle_ps(x0, x1, 0);
    tmp1 = _mm_shuffle_ps(x2, x3, 0);
    v_x0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    tmp2 = _mm_shuffle_ps(x4, x5, 0);
    tmp3 = _mm_shuffle_ps(x6, x7, 0);
    v_x1 = _mm_shuffle_ps(tmp2, tmp3, 0x88);

    v_accum0 = _mm_fmadd_ps(v_x0, v_val0, v_accum0);
    v_accum1 = _mm_fmadd_ps(v_x1, v_val1, v_accum1);

    // v_x2, v_x3
    x0 = _mm_load_ss(x + indices[8]);
    x1 = _mm_load_ss(x + indices[9]);
    x2 = _mm_load_ss(x + indices[10]);
    x3 = _mm_load_ss(x + indices[11]);
    x4 = _mm_load_ss(x + indices[12]);
    x5 = _mm_load_ss(x + indices[13]);
    x6 = _mm_load_ss(x + indices[14]);
    x7 = _mm_load_ss(x + indices[15]);

    tmp4 = _mm_shuffle_ps(x0, x1, 0);
    tmp5 = _mm_shuffle_ps(x2, x3, 0);
    v_x2 = _mm_shuffle_ps(tmp4, tmp5, 0x88);

    tmp6 = _mm_shuffle_ps(x4, x5, 0);
    tmp7 = _mm_shuffle_ps(x6, x7, 0);
    v_x3 = _mm_shuffle_ps(tmp6, tmp7, 0x88);

    v_accum0 = _mm_fmadd_ps(v_x2, v_val2, v_accum0);
    v_accum1 = _mm_fmadd_ps(v_x3, v_val3, v_accum1);

    v_y = _mm_loadu_ps(y + r);
    v_y = _mm_mul_ps(v_y, v_beta);

    v_accum = _mm_add_ps(v_accum0, v_accum1);

    tmp2 = _mm_mul_ps(v_alpha, v_accum);
    v_y = _mm_add_ps(v_y, tmp2);

    _mm_storeu_ps(y + r, v_y);

    val += 16;
    indices += 16;
  }
  // printf(" alpha %f rs %d rows is %d\n",alpha,rs,re);
  // __spmv_ell_serial_host_sse_float(alpha, beta, 0, lre, ldbin, 8,
  // bin_indices,
  //                                  bin_values, x, y);

#else
  for (ALPHA_INT r = 0; r < re; r++) {
    TYPE tmp;
    alpha_setzero(tmp);
    alpha_madde(tmp, bin_values[0], x[bin_indices[0]]);
    alpha_madde(tmp, bin_values[1], x[bin_indices[1]]);
    alpha_madde(tmp, bin_values[2], x[bin_indices[2]]);
    alpha_madde(tmp, bin_values[3], x[bin_indices[3]]);
    // printf("row %d->%d\n",r,mat->rows_indx[r]);

    y[r] = alpha_mul(y[r], beta);
    alpha_madde(y[r], tmp, alpha);

    bin_values += 4;
    bin_indices += 4;
  }
#endif
}
// bins are col-major
// ell8
inline static void bins8_mv(const TYPE alpha, const ALPHA_INT rs,
                            const ALPHA_INT *row_indx, const ALPHA_INT re,
                            ALPHA_INT *bin_indices, TYPE *bin_values,
                            const TYPE *x, TYPE *y,
                            const TYPE beta) {
#ifdef S
  ALPHA_INT lre = (re >> 2) << 2;
  ALPHA_INT ldbin = 4;
  __m128 v_alpha, v_beta;
  const TYPE *val = bin_values;
  const ALPHA_INT *indices = bin_indices;
  v_alpha = _mm_set1_ps(alpha);
  v_beta = _mm_set1_ps(beta);
  // memset(bin_indices,0,sizeof(ALPHA_INT) * re * 8);
  __m128 v_val0, v_val1, v_val2, v_val3, v_val4, v_val5, v_val6, v_val7;
  __m128 v_x0, v_x1, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7;
  __m128 x0, x1, x2, x3, x4, x5, x6, x7;
  __m128 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128 v_accum;
  __m128 v_accum0;
  __m128 v_accum1;
  __m128 v_accum2;
  __m128 v_accum3;
  __m128 v_accum4;
  __m128 v_accum5;
  __m128 v_accum6;
  __m128 v_accum7;
  int idx0, idx1, idx2, idx3;
  int idx4, idx5, idx6, idx7;
  __m128 v_y;
  // chunksize = 4
  for (ALPHA_INT r = 0; r < lre; r += 4) {
    v_accum0 = _mm_setzero_ps();
    v_accum1 = _mm_setzero_ps();

    v_val0 = _mm_loadu_ps(val + 0);
    v_val1 = _mm_loadu_ps(val + 4);
    v_val2 = _mm_loadu_ps(val + 8);
    v_val3 = _mm_loadu_ps(val + 12);
    v_val4 = _mm_loadu_ps(val + 16);
    v_val5 = _mm_loadu_ps(val + 20);
    v_val6 = _mm_loadu_ps(val + 24);
    v_val7 = _mm_loadu_ps(val + 28);

    // v_x0, v_x1
    x0 = _mm_load_ss(x + indices[0]);
    x1 = _mm_load_ss(x + indices[1]);
    x2 = _mm_load_ss(x + indices[2]);
    x3 = _mm_load_ss(x + indices[3]);
    x4 = _mm_load_ss(x + indices[4]);
    x5 = _mm_load_ss(x + indices[5]);
    x6 = _mm_load_ss(x + indices[6]);
    x7 = _mm_load_ss(x + indices[7]);

    tmp0 = _mm_shuffle_ps(x0, x1, 0);
    tmp1 = _mm_shuffle_ps(x2, x3, 0);
    v_x0 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    tmp2 = _mm_shuffle_ps(x4, x5, 0);
    tmp3 = _mm_shuffle_ps(x6, x7, 0);
    v_x1 = _mm_shuffle_ps(tmp2, tmp3, 0x88);

    v_accum0 = _mm_fmadd_ps(v_x0, v_val0, v_accum0);
    v_accum1 = _mm_fmadd_ps(v_x1, v_val1, v_accum1);

    // v_x2, v_x3
    x0 = _mm_load_ss(x + indices[8]);
    x1 = _mm_load_ss(x + indices[9]);
    x2 = _mm_load_ss(x + indices[10]);
    x3 = _mm_load_ss(x + indices[11]);
    x4 = _mm_load_ss(x + indices[12]);
    x5 = _mm_load_ss(x + indices[13]);
    x6 = _mm_load_ss(x + indices[14]);
    x7 = _mm_load_ss(x + indices[15]);

    tmp4 = _mm_shuffle_ps(x0, x1, 0);
    tmp5 = _mm_shuffle_ps(x2, x3, 0);
    v_x2 = _mm_shuffle_ps(tmp4, tmp5, 0x88);

    tmp6 = _mm_shuffle_ps(x4, x5, 0);
    tmp7 = _mm_shuffle_ps(x6, x7, 0);
    v_x3 = _mm_shuffle_ps(tmp6, tmp7, 0x88);

    v_accum0 = _mm_fmadd_ps(v_x2, v_val2, v_accum0);
    v_accum1 = _mm_fmadd_ps(v_x3, v_val3, v_accum1);
    // v_x4, v_x5
    x0 = _mm_load_ss(x + indices[16]);
    x1 = _mm_load_ss(x + indices[17]);
    x2 = _mm_load_ss(x + indices[18]);
    x3 = _mm_load_ss(x + indices[19]);
    x4 = _mm_load_ss(x + indices[20]);
    x5 = _mm_load_ss(x + indices[21]);
    x6 = _mm_load_ss(x + indices[22]);
    x7 = _mm_load_ss(x + indices[23]);

    tmp0 = _mm_shuffle_ps(x0, x1, 0);
    tmp1 = _mm_shuffle_ps(x2, x3, 0);
    v_x4 = _mm_shuffle_ps(tmp0, tmp1, 0x88);

    tmp2 = _mm_shuffle_ps(x4, x5, 0);
    tmp3 = _mm_shuffle_ps(x6, x7, 0);
    v_x5 = _mm_shuffle_ps(tmp2, tmp3, 0x88);

    v_accum0 = _mm_fmadd_ps(v_x4, v_val4, v_accum0);
    v_accum1 = _mm_fmadd_ps(v_x5, v_val5, v_accum1);

    // v_x6, v_x7
    x0 = _mm_load_ss(x + indices[24]);
    x1 = _mm_load_ss(x + indices[25]);
    x2 = _mm_load_ss(x + indices[26]);
    x3 = _mm_load_ss(x + indices[27]);
    x4 = _mm_load_ss(x + indices[28]);
    x5 = _mm_load_ss(x + indices[29]);
    x6 = _mm_load_ss(x + indices[30]);
    x7 = _mm_load_ss(x + indices[31]);

    tmp4 = _mm_shuffle_ps(x0, x1, 0);
    tmp5 = _mm_shuffle_ps(x2, x3, 0);
    v_x6 = _mm_shuffle_ps(tmp4, tmp5, 0x88);

    tmp6 = _mm_shuffle_ps(x4, x5, 0);
    tmp7 = _mm_shuffle_ps(x6, x7, 0);
    v_x7 = _mm_shuffle_ps(tmp6, tmp7, 0x88);

    v_accum0 = _mm_fmadd_ps(v_x6, v_val6, v_accum0);
    v_accum1 = _mm_fmadd_ps(v_x7, v_val7, v_accum1);

    v_y = _mm_loadu_ps(y + r);
    v_y = _mm_mul_ps(v_y, v_beta);

    v_accum = _mm_add_ps(v_accum0, v_accum1);

    tmp2 = _mm_mul_ps(v_alpha, v_accum);
    v_y = _mm_add_ps(v_y, tmp2);

    _mm_storeu_ps(y + r, v_y);

    val += 32;
    indices += 32;
  }
  // printf(" alpha %f rs %d rows is %d\n",alpha,rs,re);
  // __spmv_ell_serial_host_sse_float(alpha, beta, 0, lre, ldbin, 8,
  // bin_indices,
  //                                  bin_values, x, y);

#else
  const ALPHA_INT ldbin = re - rs;
  for (ALPHA_INT r = 0; r < re; r++) {
    TYPE tmp;
    tmp = alpha_setzero(tmp);
    tmp = alpha_madde(tmp, bin_values[0], x[bin_indices[0]]);
    tmp = alpha_madde(tmp, bin_values[1], x[bin_indices[1]]);
    tmp = alpha_madde(tmp, bin_values[2], x[bin_indices[2]]);
    tmp = alpha_madde(tmp, bin_values[3], x[bin_indices[3]]);
    tmp = alpha_madde(tmp, bin_values[4], x[bin_indices[4]]);
    tmp = alpha_madde(tmp, bin_values[5], x[bin_indices[5]]);
    tmp = alpha_madde(tmp, bin_values[6], x[bin_indices[6]]);
    tmp = alpha_madde(tmp, bin_values[7], x[bin_indices[7]]);
    // printf("row %d->%d\n",r,mat->rows_indx[r]);

    y[r] = alpha_mul(y[r], beta);
    y[r] = alpha_madde(y[r], tmp, alpha);

    bin_values += 8;
    bin_indices += 8;
  }
#endif
}

static void bins_confluence_mv(const TYPE alpha,
                               const internal_spmat mat,
                               const TYPE *x, const TYPE beta,
                               TYPE *y) {
  bins8_mv(alpha, 0, mat->rows_indx,
           mat->bins_row_end[0] - mat->bins_row_start[0],
           &mat->bins_indices[mat->bins_nz_start[0]],
           &mat->bins_values[mat->bins_nz_start[0]], x,
           y + mat->bins_row_start[0], beta);

  // printf(" row from %d to %d\n",mat->bins_row_start[1],mat->bins_row_end[1]);
  // printf(" nz from %d to %d\n",mat->bins_nz_start[1],mat->bins_nz_end[1]);

  bins4_mv(alpha, 0, mat->rows_indx,
           mat->bins_row_end[1] - mat->bins_row_start[1],
           &mat->bins_indices[mat->bins_nz_start[1]],
           &mat->bins_values[mat->bins_nz_start[1]], x,
           y + mat->bins_row_start[1], beta);

  bins2_mv(alpha, 0, mat->rows_indx,
           mat->bins_row_end[2] - mat->bins_row_start[2],
           &mat->bins_indices[mat->bins_nz_start[2]],
           &mat->bins_values[mat->bins_nz_start[2]], x,
           y + mat->bins_row_start[2], beta);

  bins1_mv(alpha, 0, mat->rows_indx,
           mat->bins_row_end[3] - mat->bins_row_start[3],
           &mat->bins_indices[mat->bins_nz_start[3]],
           &mat->bins_values[mat->bins_nz_start[3]], x,
           y + mat->bins_row_start[3], beta);

  for (ALPHA_INT r = mat->bins_row_start[4]; r < mat->bins_row_end[4]; r++) {
    y[r] = alpha_mul(y[r], beta);
  }
}

template <typename TYPE>
alphasparseStatus_t gemv_sell_csigma(const TYPE alpha,
                           const internal_spmat mat,
                           const TYPE *x, const TYPE beta,
                           TYPE *y) {
  // printf("C %d SIGMA %d\n", mat->C, mat->SIGMA);
  const ALPHA_INT num_chunks = mat->num_chunks;
  const ALPHA_INT C = mat->C;
  const ALPHA_INT rows = mat->rows;
  const ALPHA_INT *row_indices = mat->rows_indx;
  const ALPHA_INT num_threads = alpha_get_thread_num();
  ALPHA_INT partition[num_threads + 1];
  // in case the last chunk is incomplete
  TYPE *y_reordering = alpha_malloc(sizeof(TYPE) * rows);
#if 1
  if (num_chunks > 1) {
    balanced_partition_row_by_nnz(mat->chunks_end, num_chunks - 1, num_threads,
                                  partition);
    // assume C is an aliquot of rows
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
      const ALPHA_INT tid = alpha_get_thread_id();
      ALPHA_INT lcs = partition[tid];
      ALPHA_INT lce = partition[tid + 1];

      ALPHA_INT num_chunks = lce - lcs;
      ALPHA_INT row_start_chunk = lcs * C;
      ALPHA_INT row_end_chunk = lce * C;

#if DEBUG == 0 || !defined(DEBUG)
      // gather data to y_reordering
      for (ALPHA_INT r = row_start_chunk; r < row_end_chunk; r++) {
        y_reordering[r] = y[row_indices[r]];
      }
#endif

#if defined(S)

      if (C == 4)
        __spmv_sell_c4_serial_host_fma128_float(
            alpha, beta, num_chunks, mat->chunks_start + lcs, mat->indices,
            mat->values, x, y_reordering + row_start_chunk);
      else if (C == 8) {
        __spmv_sell_c8_serial_host_fma256_float(
            alpha, beta, num_chunks, mat->chunks_start + lcs, mat->indices,
            mat->values, x, y_reordering + row_start_chunk);
      } else {
        fprintf(stderr, "C %d not supported\n", C);
      }
      // scatter data to y

#elif defined(D)
      if (C == 4)
        __spmv_sell_c4_serial_host_fma256_double(
            alpha, beta, num_chunks, mat->chunks_start + lcs, mat->indices,
            mat->values, x, y_reordering + row_start_chunk);
      else if (C == 8) {
        __spmv_sell_c8_serial_host_fma256_double(
            alpha, beta, num_chunks, mat->chunks_start + lcs, mat->indices,
            mat->values, x, y_reordering + row_start_chunk);
      } else {
        fprintf(stderr, "C %d not supported dgemv\n", C);
      }
#else
      TYPE *tmp_val =
          (TYPE *)alpha_malloc(sizeof(TYPE) * C);
      for (ALPHA_INT chunk_id = lcs; chunk_id < lce; chunk_id++) {
        memset(tmp_val, 0, sizeof(TYPE) * C);
        const ALPHA_INT chunk_start = mat->chunks_start[chunk_id];
        const ALPHA_INT chunk_end = mat->chunks_end[chunk_id];
        const ALPHA_INT width = (chunk_end - chunk_start) / C;
        const TYPE *values = &mat->values[chunk_start];
        const ALPHA_INT *indices = &mat->indices[chunk_start];
        // loop on columns col - major
        for (ALPHA_INT c = 0; c < width; c++) {
          // should completely unroll here
          for (ALPHA_INT r = 0; r < C; r++) {
            tmp_val[r] = alpha_madde(tmp_val[r], values[c * C + r], x[indices[c * C + r]]);
          }
        }
        ALPHA_INT lrs = chunk_id * C;
        for (ALPHA_INT r = 0; r < C; r++, lrs++) {
          y_reordering[lrs] = alpha_mul(y_reordering[lrs], beta);
          y_reordering[lrs] = alpha_madde(y_reordering[lrs], tmp_val[r], alpha);
        }
      }
#endif

#if DEBUG == 0 || !defined(DEBUG)
      for (ALPHA_INT r = row_start_chunk; r < row_end_chunk; r++) {
        y[row_indices[r]] = y_reordering[r];
      }
#endif
    }
  }

  // the last chunk
  if (num_chunks > 0) {
    // printf("last chunk\n");
    const ALPHA_INT row_start_chunk = (num_chunks - 1) * C;
    const ALPHA_INT row_end_chunk = mat->rows_sell_end;
    const ALPHA_INT row_remains = row_end_chunk - row_start_chunk;
    const ALPHA_INT chunk_nnz =
        (mat->chunks_end[num_chunks - 1] - mat->chunks_start[num_chunks - 1]);
    const ALPHA_INT chunk_length = chunk_nnz / row_remains;
    // printf(
    //     "last chunk nnz %d len %d row_end_chunk %d row_start_chunk %d "
    //     "row_remains %d\n",
    //     chunk_nnz, chunk_length, row_end_chunk, row_start_chunk,
    //     row_remains);
    TYPE *accum =
        (TYPE *)alpha_malloc(sizeof(TYPE) * row_remains);
    memset(accum, 0, sizeof(TYPE) * row_remains);
    TYPE *val = &mat->values[mat->chunks_start[num_chunks - 1]];
    ALPHA_INT *idx = &mat->indices[mat->chunks_start[num_chunks - 1]];
    for (ALPHA_INT c = 0; c < chunk_length; c++) {
      for (ALPHA_INT r = 0; r < row_remains; r++) {
        accum[r] = alpha_madde(accum[r], val[c * row_remains + r],
                    x[idx[c * row_remains + r]]);
      }
    }
#if DEBUG == 0 || !defined(DEBUG)
    for (ALPHA_INT r = row_start_chunk; r < row_end_chunk; r++) {
      y[row_indices[r]] = alpha_mul(y[row_indices[r]], beta);
      y[row_indices[r]] = alpha_madde(y[row_indices[r]], accum[r - row_start_chunk], alpha);
    }
#else
    for (ALPHA_INT r = row_start_chunk; r < row_end_chunk; r++) {
      y_reordering[r] = alpha_mul(y_reordering[r], beta);
      y_reordering[r] = alpha_madde(y_reordering[r], accum[r - row_start_chunk], alpha);
    }
#endif
    alpha_free(accum);
  }  //! end last chunk
#endif
  //  bins not empty
  if (mat->rows_sell_end < mat->rows) {
    // printf("BINS starts from %d:%d\n", mat->bins_row_start[0],
    //        mat->bins_row_end[NUM_BINS - 1]);
#if DEBUG == 0 || !defined(DEBUG)
    for (ALPHA_INT r = mat->bins_row_start[0];
         r < mat->bins_row_end[NUM_BINS - 1]; r++) {
      y_reordering[r] = y[row_indices[r]];
    }
#endif
    bins_confluence_mv(alpha, mat, x, beta, y_reordering);

#if DEBUG == 0 || !defined(DEBUG)
    for (ALPHA_INT r = mat->bins_row_start[0];
         r < mat->bins_row_end[NUM_BINS - 1]; r++) {
      y[row_indices[r]] = y_reordering[r];
    }
#endif
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
