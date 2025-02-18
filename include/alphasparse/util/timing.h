#pragma once

/**
 * @brief header for timing utils
 */

#include <stdint.h>

typedef struct
{
    double start;
    double end;
} alpha_timer_t;

double alpha_timing_wtime();

void alpha_timing_start(alpha_timer_t *timer);

void alpha_timing_end(alpha_timer_t *timer);

double alpha_timing_elapsed_time(const alpha_timer_t *timer);

double alpha_timing_gflops(const alpha_timer_t *timer, int64_t operations);

void alpha_timing_elaped_time_print(const alpha_timer_t *timer, const char *name);

void alpha_timing_gflops_print(const alpha_timer_t *timer, int64_t operations, const char *name);

#define alpha_timing(func, ...)        \
    alpha_timer_t func##_timer;            \
    alpha_timing_start(&func##_timer); \
    func(__VA_ARGS__);               \
    alpha_timing_end(&func##_timer);   \
    alpha_timing_elaped_time_print(&func##_timer, #func);

