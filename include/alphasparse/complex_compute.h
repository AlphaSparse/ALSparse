#pragma once

#include "types.h"

#ifdef __DCU__
#include <math.h>
#endif

// #ifndef DOUBLE
// #define cmp_setzero cmp_setzerof
// #define cmp_setone cmp_setonef
// #else
// #define cmp_setzero cmp_setzerod
// #define cmp_setone cmp_setoned
// #endif
#define _iszero(real) ((0 <= real && real <= 1e-10) || (-1e-10 <= real && real <= 0))

#define cmp_iszero(val) (_iszero((val).real) && _iszero((val).imag))
#define cmp_isone(val) (_iszero((val).real - 1.f) && _iszero((val).imag))

#define cmp_conj(b, a)        \
    {                         \
        (b).real = (a).real;  \
        (b).imag = -(a).imag; \
    }

#define cmp_setzerof(z) \
    {                   \
        (z).real = 0.f; \
        (z).imag = 0.f; \
    }

#define cmp_setzerod(z) \
    {                   \
        (z).real = 0.;  \
        (z).imag = 0.;  \
    }

#define cmp_setonef(z)  \
    {                   \
        (z).real = 1.f; \
        (z).imag = 0.f; \
    }

#define cmp_setoned(z) \
    {                  \
        (z).real = 1.; \
        (z).imag = 0.; \
    }

#define cmp_add(z, x, y)                \
    {                                   \
        (z).real = (x).real + (y).real; \
        (z).imag = (x).imag + (y).imag; \
    }

#define cmp_adde(z, x)        \
    {                         \
        (z).real += (x).real; \
        (z).imag += (x).imag; \
    }

#define cmp_sub(z, x, y)                \
    {                                   \
        (z).real = (x).real - (y).real; \
        (z).imag = (x).imag - (y).imag; \
    }

#define cmp_sube(z, x)        \
    {                         \
        (z).real -= (x).real; \
        (z).imag -= (x).imag; \
    }

#define cmp_mul(z, x, y)                                                    \
    {                                                                       \
        ALPHA_Float _REAL = (x).real * (y).real - (x).imag * (y).imag;        \
        ALPHA_Float _IMAG = (x).imag * (y).real + (x).real * (y).imag;        \
        (z).real = _REAL;                                                   \
        (z).imag = _IMAG;                                                   \
    }
//z=conj(x)*y
#define cmp_mul_2c(z, x, y)                                                 \
    {                                                                       \
        ALPHA_Float _B = (x).real * (y).imag;                                 \
        ALPHA_Float _C = -(x).imag * (y).real;                                \
        (z).real = ((x).real - (x).imag) * ((y).real - (y).imag) + _B - _C; \
        (z).imag = _B + _C;                                                 \
    }

//z=x*conj(y)
#define cmp_mul_3c(z, x, y)                                                 \
    {                                                                       \
        ALPHA_Float _B = -(x).real * (y).imag;                                \
        ALPHA_Float _C = (x).imag * (y).real;                                 \
        (z).real = ((x).real + (x).imag) * ((y).real + (y).imag) + _B - _C; \
        (z).imag = _B + _C;                                                 \
    }

#define cmp_mule(z, x) cmp_mul(z, x, z)
#define cmp_mule_2c(z, x) cmp_mul_2c(z, x, z)
#define cmp_mule_3c(z, x) cmp_mul_3c(z, x, z)

#define cmp_div(result, numerator, denominator)              \
    {                                                        \
        ALPHA_Float _AC = numerator.real * denominator.real;   \
        ALPHA_Float _BD = numerator.imag * denominator.imag;   \
        ALPHA_Float _BC = numerator.imag * denominator.real;   \
        ALPHA_Float _AD = numerator.real * denominator.imag;   \
        ALPHA_Float _C2 = denominator.real * denominator.real; \
        ALPHA_Float _D2 = denominator.imag * denominator.imag; \
        result.real = (_AC + _BD) / (_C2 + _D2);             \
        result.imag = (_BC - _AD) / (_C2 + _D2);             \
    }

#define cmp_div_3c(result, numerator, denominator)           \
    {                                                        \
        ALPHA_Float _AC = numerator.real * denominator.real;   \
        ALPHA_Float _BD = -numerator.imag * denominator.imag;  \
        ALPHA_Float _BC = numerator.imag * denominator.real;   \
        ALPHA_Float _AD = -numerator.real * denominator.imag;  \
        ALPHA_Float _C2 = denominator.real * denominator.real; \
        ALPHA_Float _D2 = denominator.imag * denominator.imag; \
        result.real = (_AC + _BD) / (_C2 + _D2);             \
        result.imag = (_BC - _AD) / (_C2 + _D2);             \
    }
#define cmp_dive(result, denominator) cmp_div((result), (result), (denominator))

#ifdef __DCU__
// d = a * b + c
#define cmp_madd(d, a, b, c)                                                           \
    {                                                                                  \
        ALPHA_Float _REAL = fma(-a.imag, b.imag, fma(a.real, b.real, c.real));           \
        ALPHA_Float _IMAG = fma( a.real, b.imag, fma(a.imag, b.real, c.imag));           \
        d.real = _REAL;                                                                \
        d.imag = _IMAG;                                                                \
    }

// d = a * b + d
#define cmp_madde(d, a, b)                                                   \
    {                                                                        \
        ALPHA_Float _REAL = fma(-a.imag, b.imag, fma(a.real, b.real, d.real)); \
        ALPHA_Float _IMAG = fma( a.real, b.imag, fma(a.imag, b.real, d.imag)); \
        d.real = _REAL;                                                      \
        d.imag = _IMAG;                                                      \
    }
#else
// d = a * b + c
#define cmp_madd(d, a, b, c)                                                           \
    {                                                                                  \
        ALPHA_Float _B = (a).real * (b).imag;                                            \
        ALPHA_Float _C = (a).imag * (b).real;                                            \
        (d).real = ((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C + (c).real; \
        (d).imag = _B + _C + (c).imag;                                                 \
    }

// d = a * b + d
#define cmp_madde(d, a, b)                                                   \
    {                                                                        \
        ALPHA_Float _B = (a).real * (b).imag;                                  \
        ALPHA_Float _C = (a).imag * (b).real;                                  \
        (d).real += ((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C; \
        (d).imag += _B + _C;                                                 \
    }
#endif

// d = conj(a) * b + c
#define cmp_madd_2c(d, a, b, c)                                                        \
    {                                                                                  \
        ALPHA_Float _B = (a).real * (b).imag;                                            \
        ALPHA_Float _C = -(a).imag * (b).real;                                           \
        (d).real = ((a).real - (a).imag) * ((b).real - (b).imag) + _B - _C + (c).real; \
        (d).imag = _B + _C + (c).imag;                                                 \
    }
// d = a * conj(b) + c
#define cmp_madd_3c(d, a, b, c)                                                        \
    {                                                                                  \
        ALPHA_Float _B = -(a).real * (b).imag;                                           \
        ALPHA_Float _C = (a).imag * (b).real;                                            \
        (d).real = ((a).real + (a).imag) * ((b).real + (b).imag) + _B - _C + (c).real; \
        (d).imag = _B + _C + (c).imag;                                                 \
    }

//d += conj(a) * b
#define cmp_madde_2c(d, a, b)                                                \
    {                                                                        \
        ALPHA_Float _B = (a).real * (b).imag;                                  \
        ALPHA_Float _C = -(a).imag * (b).real;                                 \
        (d).real += ((a).real - (a).imag) * ((b).real - (b).imag) + _B - _C; \
        (d).imag += _B + _C;                                                 \
    }
//d += a * conj(b)
#define cmp_madde_3c(d, a, b)                                                \
    {                                                                        \
        ALPHA_Float _B = -(a).real * (b).imag;                                 \
        ALPHA_Float _C = (a).imag * (b).real;                                  \
        (d).real += ((a).real + (a).imag) * ((b).real + (b).imag) + _B - _C; \
        (d).imag += _B + _C;                                                 \
    }

#define cmp_msub(d, a, b, c)                                                             \
    {                                                                                    \
        ALPHA_Float _B = (a).real * (b).imag;                                              \
        ALPHA_Float _C = (a).imag * (b).real;                                              \
        (d).real = (c).real - (((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C); \
        (d).imag = (c).imag - (_B + _C);                                                 \
    }

#define cmp_msube(d, a, b)                                                   \
    {                                                                        \
        ALPHA_Float _B = (a).real * (b).imag;                                  \
        ALPHA_Float _C = (a).imag * (b).real;                                  \
        (d).real -= ((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C; \
        (d).imag -= _B + _C;                                                 \
    }

#define cmp_msube_2c(d, a, b)                                                \
    {                                                                        \
        ALPHA_Float _B = (a).real * (b).imag;                                  \
        ALPHA_Float _C = -(a).imag * (b).real;                                 \
        (d).real -= ((a).real - (a).imag) * ((b).real - (b).imag) + _B - _C; \
        (d).imag -= _B + _C;                                                 \
    }

#define cmp_copy(a, b)       \
    {                        \
        (a).real = (b).real; \
        (a).imag = (b).imag; \
    }

// flag must be 0 or 1
#define cmp_cross_entropy(a, b, c, flag) \
    {                                    \
        ALPHA_Float _t1, _t2, _t3, _t4;    \
        (_t1) = (b).real * (flag);       \
        (_t2) = (b).imag * (flag);       \
        (_t3) = (c).real * (1 - (flag)); \
        (_t4) = (c).imag * (1 - (flag)); \
        (a).real = (_t1) + (_t3);        \
        (a).imag = (_t2) + (_t4);        \
    }
