#include "alphasparse/kernel.h"

void __spmv_bsr4x1_serial_host_sse_float(const float alpha, const float beta,
                                       const int num_chunks, const int *chunks_start,
                                       const int *col_indices, const float *values,
                                       const float *x, float *y);