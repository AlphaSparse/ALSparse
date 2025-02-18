/**
 * @brief implement for timing utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include "alphasparse/util/timing.h"
#include <sys/time.h>
#include <stdint.h>

double alpha_timing_wtime()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval time;
    gettimeofday(&time, 0);
    return time.tv_sec + time.tv_usec / 1e6;
#endif
}

void alpha_timing_start(alpha_timer_t *timer)
{
    timer->start = alpha_timing_wtime();
}

void alpha_timing_end(alpha_timer_t *timer)
{
    timer->end = alpha_timing_wtime();
}

double alpha_timing_elapsed_time(const alpha_timer_t *timer)
{
    return timer->end - timer->start;
}

double alpha_timing_gflops(const alpha_timer_t *timer, int64_t operations)
{
    double opg = operations * 1e-9;
    return opg / alpha_timing_elapsed_time(timer);
}

void alpha_timing_elaped_time_print(const alpha_timer_t *timer, const char *name)
{
    printf("%s elasped time : %lf[sec]\n", name, alpha_timing_elapsed_time(timer));
}

void alpha_timing_gflops_print(const alpha_timer_t *timer, int64_t operations, const char *name)
{
    printf("%s gflops : %lf\n", name, alpha_timing_gflops(timer,operations));
}
