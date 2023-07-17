#pragma once

/**
 * @brief header for multithread utils
 */ 
#ifdef _OPENMP
#include <omp.h>
#endif

int alpha_get_core_num();

void alpha_set_thread_num(const int num);

int alpha_get_thread_num();

int alpha_get_thread_id();


