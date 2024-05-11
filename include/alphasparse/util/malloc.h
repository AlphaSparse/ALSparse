#pragma once

/**
 * @brief header for ict malloc utils
 */

#include <stdlib.h>
#include <cuda_bf16.h>
#include "../types.h"

#define DEFAULT_ALIGNMENT 32

void *alpha_malloc(size_t bytes);

void *alpha_memalign(size_t bytes, size_t alignment);

void alpha_free(void *point);

#define L1_CACHE_SIZE (64l << 10)
#define L2_CACHE_SIZE (512l << 10)
#define L3_CACHE_SIZE (32l << 20)
void alpha_clear_cache();

void alpha_fill_h(half *arr, const half num, const size_t size);
void alpha_fill_s(float *arr, const float num, const size_t size);
void alpha_fill_d(double *arr, const double num, const size_t size);
void alpha_fill_c(cuFloatComplex *arr, const cuFloatComplex num, const size_t size);
void alpha_fill_z(cuDoubleComplex *arr, const cuDoubleComplex num, const size_t size);

void alpha_parallel_fill_h(half *arr, const half num, const size_t size);
void alpha_parallel_fill_s(float *arr, const float num, const size_t size);
void alpha_parallel_fill_d(double *arr, const double num, const size_t size);
void alpha_parallel_fill_c(cuFloatComplex *arr, const cuFloatComplex num,
                         const size_t size);
void alpha_parallel_fill_z(cuDoubleComplex *arr, const cuDoubleComplex num,
                         const size_t size);

void alpha_fill_random(int8_t *arr, unsigned int seed, const size_t size);
void alpha_fill_random(int32_t *arr, unsigned int seed, const size_t size);
void alpha_fill_random(half *arr, unsigned int seed, const size_t size);
void alpha_fill_random(nv_bfloat16 *arr, unsigned int seed, const size_t size);
void alpha_fill_random(float *arr, unsigned int seed, const size_t size);
void alpha_fill_random(double *arr, unsigned int seed, const size_t size);
void alpha_fill_random(cuFloatComplex *arr, unsigned int seed, const size_t size);
void alpha_fill_random(cuDoubleComplex *arr, unsigned int seed,
                       const size_t size);
void alpha_fill_random(half2 *arr, unsigned int seed, const size_t size);
void alpha_fill_random(nv_bfloat162 *arr, unsigned int seed, const size_t size);
void alpha_parallel_fill_random_h(half *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_s(float *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_d(double *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_c(cuFloatComplex *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_z(cuDoubleComplex *arr, unsigned int seed,
                                const size_t size);

void alpha_fill_random(int *arr, unsigned int seed, const size_t size,
                         int upper);

void alpha_fill_random(long long *arr, unsigned int seed, const size_t size,
                          long long upper);
