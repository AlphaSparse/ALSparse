#include <immintrin.h>
#include <stdio.h>
#include <string.h>

void __spmv_ell_serial_host_u4_float(const float alpha, const float beta,
                                     const int lrs, const int lre, const int m,
                                     const int width, const int *indices,
                                     const float *values, const float *x,
                                     float *y) {
  for (int r = lrs; r < lre; r += 4) {
    float tmp[4] = {0, 0, 0, 0};
    for (int c = 0; c < width; c++) {
      float val0 = values[c * m + r + 0];
      float val1 = values[c * m + r + 1];
      float val2 = values[c * m + r + 2];
      float val3 = values[c * m + r + 3];
      int idx0 = indices[c * m + r + 0];
      int idx1 = indices[c * m + r + 1];
      int idx2 = indices[c * m + r + 2];
      int idx3 = indices[c * m + r + 3];
      tmp[0] += x[idx0] * val0;
      tmp[1] += x[idx1] * val1;
      tmp[2] += x[idx2] * val2;
      tmp[3] += x[idx3] * val3;
    }
    y[r + 0] *= beta;
    y[r + 1] *= beta;
    y[r + 2] *= beta;
    y[r + 3] *= beta;
    y[r + 0] += tmp[0] * alpha;
    y[r + 1] += tmp[1] * alpha;
    y[r + 2] += tmp[2] * alpha;
    y[r + 3] += tmp[3] * alpha;
  }
}

void __spmv_ell_serial_host_plain_float(const float alpha, const float beta,
                                        const int lrs, const int lre,
                                        const int m, const int width,
                                        const int *indices, const float *values,
                                        const float *x, float *y) {
  for (int r = lrs; r < lre; r++) {
    float tmp = 0;
    for (int c = 0; c < width; c++) {
      float val0 = values[c * m + r + 0];
      int idx0 = indices[c * m + r + 0];
      tmp += x[idx0] * val0;
    }
    y[r + 0] *= beta;
    y[r + 0] += tmp * alpha;
  }
}

// void __spmv_ell_serial_host_sse_float(const float alpha, const float beta,
//                                       const int lrs, const int lre, const int
//                                       m, const int width, const int *indices,
//                                       const float *values, const float *x,
//                                       float *y) {
//   __m128 v_val, v_x;
//   __m128 v_alpha, v_beta;
//   __m128 v_x0, v_x1, v_x2, v_x3;
//   __m128 tmp0, tmp1;
//   __m128 v_y;
//   __m128 v_accum;
//   v_alpha = _mm_set1_ps(alpha);
//   v_beta = _mm_set1_ps(beta);
//   for (int r = lrs; r < lre; r += 4) {
//     float tmp[4] = {0, 0, 0, 0};
//     v_accum = _mm_setzero_ps();
//     for (int c = 0; c < width; c++) {
//       v_val = _mm_loadu_ps(values + c * m + r);
//       int idx0 = indices[c * m + r + 0];
//       int idx1 = indices[c * m + r + 1];
//       int idx2 = indices[c * m + r + 2];
//       int idx3 = indices[c * m + r + 3];
//       v_x0 = _mm_load_ss(x + idx0);
//       v_x1 = _mm_load_ss(x + idx1);
//       v_x2 = _mm_load_ss(x + idx2);
//       v_x3 = _mm_load_ss(x + idx3);
//       tmp0 = _mm_shuffle_ps(v_x0, v_x1, 0);
//       tmp1 = _mm_shuffle_ps(v_x2, v_x3, 0);
//       v_x = _mm_shuffle_ps(tmp0, tmp1, 0x88);
//       v_accum = _mm_fmadd_ps(v_x, v_val, v_accum);
//     }
//     v_y = _mm_loadu_ps(y + r);
//     v_y = _mm_mul_ps(v_y, v_beta);
//     v_accum = _mm_fmadd_ps(v_alpha, v_accum, v_y);
//     _mm_storeu_ps(y + r, v_accum);
//   }
// }
// void __spmv_sell_c4_serial_host_sse_float(const float alpha, const float beta,
//                                           const int num_chunks,
//                                           const int *chunks_start,
//                                           const int *col_indices,
//                                           const float *values, const float *x,
//                                           float *y)

// {
//   __m128 v_val, v_x;
//   __m128 v_alpha, v_beta;
//   __m128 v_x0, v_x1, v_x2, v_x3;
//   __m128 tmp0, tmp1;
//   __m128 v_y;
//   __m128 v_accum;
//   v_alpha = _mm_set1_ps(alpha);
//   v_beta = _mm_set1_ps(beta);
//   for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
//     const int chunk_start = chunks_start[chunk_id];
//     const int chunk_end = chunks_start[chunk_id + 1];
//     const int width = (chunk_end - chunk_start) >> 2;
//     const float *VAL = &values[chunk_start];
//     const int *indices = &col_indices[chunk_start];
//     const int lrs = chunk_id * 4;
//     v_accum = _mm_setzero_ps();
//     for (int c = 0; c < width; c++) {
//       v_val = _mm_loadu_ps(VAL + c * 4);
//       int idx0 = indices[c * 4 + 0];
//       int idx1 = indices[c * 4 + 1];
//       int idx2 = indices[c * 4 + 2];
//       int idx3 = indices[c * 4 + 3];
//       v_x0 = _mm_load_ss(x + idx0);
//       v_x1 = _mm_load_ss(x + idx1);
//       v_x2 = _mm_load_ss(x + idx2);
//       v_x3 = _mm_load_ss(x + idx3);
//       tmp0 = _mm_shuffle_ps(v_x0, v_x1, 0);
//       tmp1 = _mm_shuffle_ps(v_x2, v_x3, 0);
//       v_x = _mm_shuffle_ps(tmp0, tmp1, 0x88);
//       v_accum = _mm_fmadd_ps(v_x, v_val, v_accum);
//     }
//     v_y = _mm_loadu_ps(y + lrs);
//     v_y = _mm_mul_ps(v_y, v_beta);
//     v_accum = _mm_fmadd_ps(v_alpha, v_accum, v_y);
//     _mm_storeu_ps(y + lrs, v_accum);
//   }
// }