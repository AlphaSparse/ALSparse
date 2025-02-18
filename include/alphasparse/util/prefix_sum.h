#pragma once
#include <stdint.h>
#include <stdlib.h>
typedef enum {
  INC_SCAN = 0,
  EXL_SCAN = 1,
} prefix_sum_type_t;
// support inplace prefix_sum
void prefix_sum(prefix_sum_type_t scan_type, const int32_t* source, const size_t len,
                int32_t* output);