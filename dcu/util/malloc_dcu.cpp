#include "alphasparse/util/malloc.h"

#include <malloc.h>
#include <stdio.h>
#include <time.h>
#include <hip/hip_runtime_api.h>

void alpha_free_dcu(void *point)
{
    if (point) {
#ifdef __DCU__
        hipFree(point);
#else
        alpha_free(point);
#endif
    }
}
