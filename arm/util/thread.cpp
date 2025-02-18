/**
 * @brief implement for multithread utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util/thread.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int _thread_num;

int alpha_get_core_num()
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

void alpha_set_thread_num(const int thread_num)
{
#ifdef _OPENMP
    _thread_num = thread_num;
#else
    _thread_num = 1;
#endif
}

int alpha_get_thread_num()
{
#ifdef _OPENMP
    char * omp_thread_str =  getenv ("OMP_NUM_THREADS");
    if(omp_thread_str){
      _thread_num = atoi(omp_thread_str);
    }
    return _thread_num == 0 ? alpha_get_core_num() : _thread_num;
#else
    return 1;
#endif
}

int alpha_get_thread_id()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}