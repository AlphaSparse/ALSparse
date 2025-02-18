#pragma once

#include <stdlib.h>
#include <time.h>
#ifdef __CUDA__
#include <cuda_fp16.h>
#include <cuda_bf16.hpp>
#endif
inline unsigned int time_seed() {
  time_t t;
  return (unsigned)time(&t);
}

inline int random_int(int m) { return rand() % m; }
inline long long random_long(long long m) { return (long long)rand() % m; }

inline double random_double() { return (double)rand() / RAND_MAX; }

inline float random_float() { return (float)rand() / RAND_MAX; }

#ifdef __CUDA__
inline half random_half() { return half((float)rand() / RAND_MAX); }

inline nv_bfloat16 random_bf16() { return nv_bfloat16((float)rand() / RAND_MAX); }
#endif
