#pragma once
#include <stdlib.h>
typedef struct {
  int base;
  unsigned char* data;
  int sft;
  size_t num_of_char;
  size_t num_elements;
  size_t num_of_one;
} bitmap_t;

void bitmap_build(bitmap_t** bitmap, int num_elements);
void bitmap_destory(bitmap_t* bitmap);
void set_bit(bitmap_t* bitmap, int index);
void set_bit_batch(bitmap_t* bitmap, const int* index, const int batch_size);
void set_bytes(bitmap_t* bitmap, unsigned char byte, int index);
void set_bytes_batch(bitmap_t* bitmap, unsigned char byte, int index);
void set_bit_batch_sht_index(bitmap_t* bitmap, const int* index, const int batch_size,
                             const int index_sft);
int set_clear_bit_batch_sht_index(bitmap_t* bitmap, const int* index, const int batch_size,
                                  const int index_sft);
void set_bit_batch_scale_index(bitmap_t* bitmap, const int* index, const int batch_size,
                               const int index_scale);
int set_clear_bit_batch_scale_index(bitmap_t* bitmap, const int* index, const int batch_size,
                                    const int index_scale);
void clear_bit(bitmap_t* bitmap, int index);
void clear_bit_batch(bitmap_t* bitmap, const int* index, const int batch_size);
void clear_bit_batch_sht_index(bitmap_t* bitmap, const int* index, const int batch_size,
                               const int index_sft);
void clear_bit_batch_scale_index(bitmap_t* bitmap, const int* index, const int batch_size,
                                 const int index_scale);
void clear_bytes(bitmap_t* bitmap, int index);

unsigned char get_bit(bitmap_t* bitmap, int index);
unsigned char get_bytes(bitmap_t* bitmap, int index);

void bitmap_reset(bitmap_t* bitmap);
void bitmap_and(const bitmap_t* bitmap_a, const bitmap_t* bitmap_b, bitmap_t* bitmap_c);
void bitmap_or(const bitmap_t* bitmap_a, const bitmap_t* bitmap_b, bitmap_t* bitmap_c);
