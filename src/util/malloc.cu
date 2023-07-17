/**
 * @brief implement for malloc utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util/malloc.h"

#include <malloc.h>
#include <stdio.h>
#include <time.h>

#include "alphasparse/util/random.h"

#ifdef NUMA
#include <numa.h>
#endif

void *alpha_malloc(size_t bytes) {
#ifdef NUMA
  void *ret = numa_alloc_onnode(bytes, 0);
#else
  void *ret = malloc(bytes);
#endif
  if (ret == NULL) {
    printf("no enough memory space to alloc!!!\n");
    exit(-1);
  }
  return ret;
}

void *alpha_memalign(size_t bytes, size_t alignment) {
#ifdef NUMA
  void *ret = numa_alloc_onnode(bytes, 0);
#else
  void *ret = memalign(alignment, bytes);
#endif
  if (ret == NULL) {
    printf("no enough memory space to alloc!!!");
    exit(-1);
  }
  return ret;
}

void alpha_free(void *point) { 
  if (!point)
    free(point); 
}

void alpha_fill_s(float *arr, const float num, const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_d(double *arr, const double num, const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_c(cuFloatComplex *arr, const cuFloatComplex num, const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_z(cuDoubleComplex *arr, const cuDoubleComplex num,
                const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}


void alpha_fill_random(int *arr, unsigned int seed, const size_t size,
                         int upper) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_int(upper);
}
void alpha_fill_random(long long *arr, unsigned int seed, const size_t size,
                          long long upper) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_long(upper);
}

void alpha_fill_random(nv_bfloat16 *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_bf16();
}

void alpha_fill_random(half *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_half();
}

void alpha_fill_random(int8_t *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_int(3);
}

void alpha_fill_random(int32_t *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_int(3);
}

void alpha_fill_random(float *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_float();
}

void alpha_fill_random(double *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_double();
}

void alpha_fill_random(cuFloatComplex *arr, unsigned int seed,
                       const size_t size) {
  alpha_fill_random((float *)arr, seed, size * 2);
}

void alpha_fill_random(cuDoubleComplex *arr, unsigned int seed,
                       const size_t size) {
  alpha_fill_random((double *)arr, seed, size * 2);
}

void alpha_fill_random(half2 *arr, unsigned int seed, const size_t size){
  alpha_fill_random((half *)arr, seed, size * 2);
}

void alpha_fill_random(nv_bfloat162 *arr, unsigned int seed, const size_t size){
  alpha_fill_random((nv_bfloat16*)arr, seed, size * 2);
}

void alpha_parallel_fill_random_c(cuFloatComplex *arr, unsigned int seed,
                                const size_t size) {
  alpha_fill_random((float *)arr, seed, size * 2);
}

void alpha_parallel_fill_random_z(cuDoubleComplex *arr, unsigned int seed,
                                const size_t size) {
  alpha_fill_random((double *)arr, seed, size * 2);
}
