#include <immintrin.h>
#include <stdio.h>
#include <string.h>

// void __spmv_bsr4x1_serial_host_sse_float(const float alpha, const float beta,
//                                          const int num_chunks,
//                                          const int *chunks_start,
//                                          const int *col_indices,
//                                          const float *values, const float *x,
//                                          float *y) {
//   for (int i = 0, j = 0; i < num_chunks; i++, j += 4) {
//     float tmp[4] = {0, 0, 0, 0};
//     for (int ai = chunks_start[i]; ai < chunks_start[i + 1]; ai++) {
//       const float *val = &values[ai << 2];
//       const float *rhs = &x[col_indices[ai]];
//       tmp[0] += val[0] * rhs[0];
//       tmp[1] += val[1] * rhs[0];
//       tmp[2] += val[2] * rhs[0];
//       tmp[3] += val[3] * rhs[0];
//     }
//     y[j + 0] *= beta;
//     y[j + 1] *= beta;
//     y[j + 2] *= beta;
//     y[j + 3] *= beta;
//     y[j + 0] += tmp[0] * alpha;
//     y[j + 1] += tmp[1] * alpha;
//     y[j + 2] += tmp[2] * alpha;
//     y[j + 3] += tmp[3] * alpha;
//   }
// }