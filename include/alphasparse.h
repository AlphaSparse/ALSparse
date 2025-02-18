#pragma once

#include <fstream>

// #include "alphasparse/types.h"        // basic type define
// #ifdef __CUDA__
// #include <cuComplex.h>
// #endif
// #ifdef __HIP__
// #include <hip/hip_cooperative_groups.h>
// #endif
#include "alphasparse/handle.h"  // handle
#include "alphasparse/spapi.h"        // spblas API
#include "alphasparse/spdef.h"        // spblas type define
#include "alphasparse/util.h"
#include "alphasparse/kernel.h"

#if (!defined(__HYGON__)) && (!defined(__ARM__))
#include "alphasparse/common.h"
#endif
#if defined(__HYGON__) || defined(__PLAIN__) || defined(__ARM__)
#include "alphasparse/util/timing.h"
#endif
#include "alphasparse/util/internal_check.h"
#include "alphasparse/util/error.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/util/auxiliary.h"
#include "alphasparse/util/thread.h"

#include "alphasparse/spapi_plain.h"  // spblas plain API
#ifdef __HIP__
#include "alphasparse/spapi_dcu.h"  // spblas API for DCU
#endif
#ifdef __DCU__
#include "alphasparse/spapi_dcu.h"  // spblas API for DCU
#endif