#include "alphasparse/util/bitmap.h"

#include <stdio.h>
#include <string.h>

#include "alphasparse/util/malloc.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif
static int64_t bits_table_char[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

void bitmap_build(bitmap_t** bitmap, int num_of_elements) {
  bitmap_t* temp_bitmap = (bitmap_t*)alpha_malloc(sizeof(bitmap_t));
  *bitmap = temp_bitmap;
  // char is 3
  const int sft = 3;
  const size_t num_char = (num_of_elements + 8 * sizeof(char) - 1) >> sft;
  char* container = (char*)alpha_malloc(sizeof(char) * num_char);
  memset(container, 0, sizeof(char) * num_char);
  temp_bitmap->sft = sft;
  temp_bitmap->data = (unsigned char*)container;
  temp_bitmap->num_of_char = num_char;
  temp_bitmap->num_of_one = 0;
  temp_bitmap->num_elements = num_of_elements;
}

void bitmap_destory(bitmap_t* bitmap) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  alpha_free(bitmap->data);
}
void set_bit(bitmap_t* bitmap, int index) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  int char_indx = index >> bitmap->sft;
  // mod
  const unsigned char to_set = (1 << (index & 7)) | bitmap->data[char_indx];
  bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
  bitmap->data[char_indx] = to_set;
  bitmap->num_of_one += bits_table_char[(int)to_set];
}
void set_bit_batch(bitmap_t* bitmap, const int* index, const int batch_size) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }

  int i = 0;
  const int sft = bitmap->sft;
  // #ifdef __aarch64__
  //   uint32x4_t v_seven, v_8x16_3;
  //   uint32x4_t v_32x4_0, v_32x4_1, v_32x4_2, v_32x4_3;
  //   uint32x4_t v_char_offset32_0, v_char_offset32_1, v_char_offset32_2, v_char_offset32_3;
  //   uint16x4_t v_char_offset16_0, v_char_offset16_1, v_char_offset16_2, v_char_offset16_3;
  //   v_seven = vdupq_n_u32(7);
  //   unsigned int char_loc[16];
  //   unsigned int char_offset[16];
  //   unsigned char to_set[16];
  //   // process 16 chars per iteration
  //   for (; i < batch_size - 15; i += 16) {
  //     v_32x4_0 = vld1q_u32((void*)(index + i));
  //     v_32x4_1 = vld1q_u32((void*)(index + i + 4));
  //     v_32x4_2 = vld1q_u32((void*)(index + i + 8));
  //     v_32x4_3 = vld1q_u32((void*)(index + i + 12));
  //     v_char_offset32_0 = vandq_u32(v_32x4_0, v_seven);
  //     v_char_offset32_1 = vandq_u32(v_32x4_1, v_seven);
  //     v_char_offset32_2 = vandq_u32(v_32x4_2, v_seven);
  //     v_char_offset32_3 = vandq_u32(v_32x4_3, v_seven);

  //     vst1q_u32(char_offset + 0, v_char_offset32_0);
  //     vst1q_u32(char_offset + 4, v_char_offset32_1);
  //     vst1q_u32(char_offset + 8, v_char_offset32_2);
  //     vst1q_u32(char_offset + 12, v_char_offset32_3);

  //     v_32x4_0 = vshrq_n_u32(v_32x4_0, sft);
  //     v_32x4_1 = vshrq_n_u32(v_32x4_1, sft);
  //     v_32x4_2 = vshrq_n_u32(v_32x4_2, sft);
  //     v_32x4_3 = vshrq_n_u32(v_32x4_3, sft);

  //     vst1q_u32(char_loc + 0, v_32x4_0);
  //     vst1q_u32(char_loc + 4, v_32x4_1);
  //     vst1q_u32(char_loc + 8, v_32x4_2);
  //     vst1q_u32(char_loc + 12, v_32x4_3);

  //     to_set[0] = 1 << char_offset[0];
  //     to_set[1] = 1 << char_offset[1];
  //     to_set[2] = 1 << char_offset[2];
  //     to_set[3] = 1 << char_offset[3];
  //     to_set[4] = 1 << char_offset[4];
  //     to_set[5] = 1 << char_offset[5];
  //     to_set[6] = 1 << char_offset[6];
  //     to_set[7] = 1 << char_offset[7];

  //     to_set[8] = 1 << char_offset[8];
  //     to_set[9] = 1 << char_offset[9];
  //     to_set[10] = 1 << char_offset[10];
  //     to_set[11] = 1 << char_offset[11];
  //     to_set[12] = 1 << char_offset[12];
  //     to_set[13] = 1 << char_offset[13];
  //     to_set[14] = 1 << char_offset[14];
  //     to_set[15] = 1 << char_offset[15];
  //   }
  // #endif
  for (; i < batch_size; i++) {
    int char_indx = index[i] >> bitmap->sft;
    // mod
    const unsigned char to_set = (1 << (index[i] & 7)) | bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }
}
void clear_bit_batch(bitmap_t* bitmap, const int* index, const int batch_size) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }

  int i = 0;
  const int sft = bitmap->sft;
  // #ifdef __aarch64__
  //   uint32x4_t v_seven, v_8x16_3;
  //   uint32x4_t v_32x4_0, v_32x4_1, v_32x4_2, v_32x4_3;
  //   uint32x4_t v_char_offset32_0, v_char_offset32_1, v_char_offset32_2, v_char_offset32_3;
  //   uint16x4_t v_char_offset16_0, v_char_offset16_1, v_char_offset16_2, v_char_offset16_3;
  //   v_seven = vdupq_n_u32(7);
  //   unsigned int char_loc[16];
  //   unsigned int char_offset[16];
  //   unsigned char to_set[16];
  //   // process 16 chars per iteration
  //   for (; i < batch_size - 15; i += 16) {
  //     v_32x4_0 = vld1q_u32((void*)(index + i));
  //     v_32x4_1 = vld1q_u32((void*)(index + i + 4));
  //     v_32x4_2 = vld1q_u32((void*)(index + i + 8));
  //     v_32x4_3 = vld1q_u32((void*)(index + i + 12));
  //     v_char_offset32_0 = vandq_u32(v_32x4_0, v_seven);
  //     v_char_offset32_1 = vandq_u32(v_32x4_1, v_seven);
  //     v_char_offset32_2 = vandq_u32(v_32x4_2, v_seven);
  //     v_char_offset32_3 = vandq_u32(v_32x4_3, v_seven);

  //     vst1q_u32(char_offset + 0, v_char_offset32_0);
  //     vst1q_u32(char_offset + 4, v_char_offset32_1);
  //     vst1q_u32(char_offset + 8, v_char_offset32_2);
  //     vst1q_u32(char_offset + 12, v_char_offset32_3);

  //     v_32x4_0 = vshrq_n_u32(v_32x4_0, sft);
  //     v_32x4_1 = vshrq_n_u32(v_32x4_1, sft);
  //     v_32x4_2 = vshrq_n_u32(v_32x4_2, sft);
  //     v_32x4_3 = vshrq_n_u32(v_32x4_3, sft);

  //     vst1q_u32(char_loc + 0, v_32x4_0);
  //     vst1q_u32(char_loc + 4, v_32x4_1);
  //     vst1q_u32(char_loc + 8, v_32x4_2);
  //     vst1q_u32(char_loc + 12, v_32x4_3);

  //     to_set[0] = 1 << char_offset[0];
  //     to_set[1] = 1 << char_offset[1];
  //     to_set[2] = 1 << char_offset[2];
  //     to_set[3] = 1 << char_offset[3];
  //     to_set[4] = 1 << char_offset[4];
  //     to_set[5] = 1 << char_offset[5];
  //     to_set[6] = 1 << char_offset[6];
  //     to_set[7] = 1 << char_offset[7];

  //     to_set[8] = 1 << char_offset[8];
  //     to_set[9] = 1 << char_offset[9];
  //     to_set[10] = 1 << char_offset[10];
  //     to_set[11] = 1 << char_offset[11];
  //     to_set[12] = 1 << char_offset[12];
  //     to_set[13] = 1 << char_offset[13];
  //     to_set[14] = 1 << char_offset[14];
  //     to_set[15] = 1 << char_offset[15];
  //   }
  // #endif
  for (; i < batch_size; i++) {
    int char_indx = index[i] >> bitmap->sft;
    // mod
    const unsigned char to_set = ~(1 << (index[i] & 7)) & bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }
}

void set_bit_batch_sht_index(bitmap_t* bitmap, const int* index, const int batch_size,
                             const int index_sft) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }

  int i = 0;
  const int sft = bitmap->sft;
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] >> index_sft;
    int char_indx = idx >> sft;
    const unsigned char to_set = (1 << (idx & 7)) | bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }
}
void clear_bit_batch_sht_index(bitmap_t* bitmap, const int* index, const int batch_size,
                               const int index_sft) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }

  int i = 0;
  const int sft = bitmap->sft;
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] >> index_sft;
    int char_indx = idx >> sft;
    const unsigned char to_set = ~(1 << (idx & 7)) & bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }
}

// only for counting bit, not affect the bitmap itself
int set_clear_bit_batch_sht_index(bitmap_t* bitmap, const int* index, const int batch_size,
                                  const int index_sft) {
  int i = 0;
  int res = bitmap->num_of_one;
  int delta_bit = 0;
  const int sft = bitmap->sft;
  // set
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] >> index_sft;
    int char_indx = idx >> sft;
    const unsigned char to_set = (1 << (idx & 7)) | bitmap->data[char_indx];
    // printf("idx %d, char_index %d, ori char %x, char2set %x, before %ld, minus %ld, after %ld,",
    //        idx, char_indx, bitmap->data[char_indx], to_set, bitmap->num_of_one,
    //        bits_table_char[(int)bitmap->data[char_indx]],
    //        bits_table_char[(int)bitmap->data[char_indx]]);
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
    // printf("add %ld, after %ld\n", bits_table_char[(int)to_set], bitmap->num_of_one);
  }

  res = bitmap->num_of_one - res;
  i = 0;
  // clear
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] >> index_sft;
    int char_indx = idx >> sft;
    const unsigned char to_set = (~(1 << (idx & 7))) & bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }

  return res;
}

void set_bit_batch_scale_index(bitmap_t* bitmap, const int* index, const int batch_size,
                               const int index_scale) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }

  int i = 0;
  const int sft = bitmap->sft;
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] / index_scale;
    int char_indx = idx >> sft;
    const unsigned char to_set = (1 << (idx & 7)) | bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }
}
void clear_bit_batch_scale_index(bitmap_t* bitmap, const int* index, const int batch_size,
                                 const int index_scale) {
  int i = 0;
  const int sft = bitmap->sft;
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] / index_scale;
    int char_indx = idx >> sft;
    const unsigned char to_set = ~(1 << (idx & 7)) & bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }
}

// only for counting bit, not affect the bitmap itself
int set_clear_bit_batch_scale_index(bitmap_t* bitmap, const int* index, const int batch_size,
                                    const int index_scale) {
  int i = 0;
  int res = bitmap->num_of_one;
  int delta_bit = 0;
  const int sft = bitmap->sft;

  // set
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] / index_scale;
    int char_indx = idx >> sft;
    const unsigned char to_set = (1 << (idx & 7)) | bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }

  res = bitmap->num_of_one - res;
  i = 0;
  // clear
  for (; i < batch_size; i++) {
    // mod
    int idx = index[i] / index_scale;
    int char_indx = idx >> sft;
    const unsigned char to_set = (~(1 << (idx & 7))) & bitmap->data[char_indx];
    bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
    bitmap->data[char_indx] = to_set;
    bitmap->num_of_one += bits_table_char[(int)to_set];
  }

  return res;
}

void set_bytes(bitmap_t* bitmap, unsigned char byte, int index) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  // printf("before set byte %d has %ld , minus byte is %d \n", byte, bitmap->num_of_one,
  //        (int)bitmap->data[index]);
  bitmap->num_of_one -= bits_table_char[(int)bitmap->data[index]];
  bitmap->data[index] = byte;
  bitmap->num_of_one += bits_table_char[(int)byte];
}

void clear_bit(bitmap_t* bitmap, int index) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  int char_indx = index >> bitmap->sft;
  // mod
  unsigned char to_set = (~(1 << (index & 7))) & bitmap->data[char_indx];
  bitmap->num_of_one -= bits_table_char[(int)bitmap->data[char_indx]];
  bitmap->data[char_indx] = to_set;
  bitmap->num_of_one += bits_table_char[(int)to_set];
}
void clear_bytes(bitmap_t* bitmap, int index) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  bitmap->num_of_one -= bits_table_char[(int)bitmap->data[index]];
  bitmap->data[index] = 0;
}

unsigned char get_bit(bitmap_t* bitmap, int index) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  int char_indx = index >> bitmap->sft;
  const char res = (1 << (index & 7)) & bitmap->data[char_indx];
  return res;
}
unsigned char get_bytes(bitmap_t* bitmap, int index) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  return bitmap->data[index];
}

void bitmap_reset(bitmap_t* bitmap) {
  // if (!bitmap) {
  //     printf("input bitmap is null\n");
  //     exit(-1);
  //   }
  //   if (!bitmap->data) {
  //     printf("input bitmap is not initialized\n");
  //     exit(-1);
  //   }
  bitmap->num_of_one = 0;
  memset(bitmap->data, 0, sizeof(char) * bitmap->num_of_char);
}

void bitmap_and(const bitmap_t* bitmap_a, const bitmap_t* bitmap_b, bitmap_t* bitmap_c) {
  if (!bitmap_a || !bitmap_b || !bitmap_c) {
    printf("input bitmap is null\n");
    exit(-1);
  }
  if (!bitmap_a->data || !bitmap_b->data) {
    printf("input bitmap is not initialized\n");
    exit(-1);
  }
  if (bitmap_a->num_of_char != bitmap_b->num_of_char ||
      bitmap_c->num_of_char != bitmap_b->num_of_char ||
      bitmap_a->num_of_char != bitmap_c->num_of_char) {
    printf("length is not the same\n");
    exit(-1);
  }
  size_t num_char = bitmap_b->num_of_char;
  for (size_t i = 0; i < bitmap_b->num_of_char; i++) {
    char c = bitmap_c->data[i];
    bitmap_c->num_of_one -= bits_table_char[(int)c];
    c = bitmap_a->data[i] & bitmap_b->data[i];
    bitmap_c->data[i] = c;
    bitmap_c->num_of_one += bits_table_char[(int)c];
  }
}
void bitmap_or(const bitmap_t* bitmap_a, const bitmap_t* bitmap_b, bitmap_t* bitmap_c) {
  if (!bitmap_a || !bitmap_b || !bitmap_c) {
    printf("input bitmap is null\n");
    exit(-1);
  }
  if (!bitmap_a->data || !bitmap_b->data) {
    printf("input bitmap is not initialized\n");
    exit(-1);
  }
  if (bitmap_a->num_of_char != bitmap_b->num_of_char ||
      bitmap_c->num_of_char != bitmap_b->num_of_char ||
      bitmap_a->num_of_char != bitmap_c->num_of_char) {
    printf("length is not the same\n");
    exit(-1);
  }
  size_t num_char = bitmap_b->num_of_char;
  for (size_t i = 0; i < bitmap_b->num_of_char; i++) {
    char c = bitmap_c->data[i];
    bitmap_c->num_of_one -= bits_table_char[(int)c];
    c = bitmap_a->data[i] | bitmap_b->data[i];
    bitmap_c->data[i] = c;
    bitmap_c->num_of_one += bits_table_char[(int)c];
  }
}