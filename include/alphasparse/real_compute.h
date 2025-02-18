#pragma once

#ifdef __DCU__
#include <math.h>
#endif

#include "types.h"

// #ifndef DOUBLE
// #define real_setzero real_setzerof
// #define real_setone  real_setonef
// #else
// #define real_setzero real_setzerof
// #define real_setone  real_setonef
// #endif
#define real_iszero(val) ((0 <= (val) && (val) <= 1e-10) || (-1e-10 <= (val) && (val) <= 0))
#define real_isone(val) real_iszero((val)-1.f)

#define real_setzerof(a) ((a) = 0.f)
#define real_setzerod(a) ((a) = 0.)

#define real_setonef(a) ((a) = 1.f)
#define real_setoned(a) ((a) = 1.)

#define real_add(c, a, b) ((c) = (a) + (b))
#define real_adde(c, b) ((c) += (b))

#define real_sub(c, a, b) ((c) = (a) - (b))
#define real_sube(c, b) ((c) -= (b))

#define real_mul(c, a, b) ((c) = (a) * (b))
#define real_mule(c, b) ((c) *= (b))

#define real_div(c, a, b) ((c) = (a) / (b))
#define real_dive(c, b) ((c) /= (b))

#ifdef __DCU__
#define real_madd(d, a, b, c) ((d) = fma((a), (b), (c)))
#define real_madde(d, a, b) ((d) = fma((a), (b), (d)))
#else
#define real_madd(d, a, b, c) ((d) = (a) * (b) + (c))
#define real_madde(d, a, b) ((d) += (a) * (b))
#endif

#define real_msub(d, a, b, c) ((d) = (c) - (a) * (b))
#define real_msube(d, a, b) ((d) -= (a) * (b))

#define real_copy(a, b) ((a) = (b))
// a = flag * b + (1 - flag) * c
#define real_cross_entropy(a, b, c, flag) ((a) = (flag) * (b) + (1 - (flag)) * (c))
