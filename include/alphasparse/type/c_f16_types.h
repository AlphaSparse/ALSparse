/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef C_F16_TYPES_H
#define C_F16_TYPES_H

// #define __PRINT_MACRO(x) #x
// #define PRINT_MARCO(x) #x"=" __PRINT_MACRO(x)
// #pragma message(PRINT_MARCO(__CUDA_ARCH__))

#include "c_f32_types.h"
#include <cuComplex.h>
#include <cuda_fp16.h>

__host__ __device__ static __inline__ half
real_c_f16(half2 x)
{
  return x.x;
}

__host__ __device__ static __inline__ half
imag_c_f16(half2 x)
{
  return x.y;
}

__device__ static __inline__ half2
conj_c_f16(half2 x)
{
  return make_half2(real_c_f16(x), -x.y);
}

__device__ static __inline__ half2
add_c_f16(half2 x, half2 y)
{
  return make_half2(real_c_f16(x) + real_c_f16(y),
                    imag_c_f16(x) + imag_c_f16(y));
}

__device__ static __inline__ half2
sub_c_f16(half2 x, half2 y)
{
  return make_half2(real_c_f16(x) - real_c_f16(y),
                    imag_c_f16(x) - imag_c_f16(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we
 * don't do it either to stay competitive.
 */
__device__ static __inline__ half2
mul_c_f16(half2 x, half2 y)
{
  half2 prod;
  prod = make_half2(
    (real_c_f16(x) * real_c_f16(y)) - (imag_c_f16(x) * imag_c_f16(y)),
    (real_c_f16(x) * imag_c_f16(y)) + (imag_c_f16(x) * real_c_f16(y)));
  return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
__device__ static __inline__ half2
div_c_f16(half2 x, half2 y)
{
  half2 quot;
  half s = (__habs(real_c_f16(y))) + (__habs(imag_c_f16(y)));
  half oos = (half)1.0 / s;
  half ars = real_c_f16(x) * oos;
  half ais = imag_c_f16(x) * oos;
  half brs = real_c_f16(y) * oos;
  half bis = imag_c_f16(y) * oos;
  s = (brs * brs) + (bis * bis);
  oos = (half)1.0 / s;
  quot = make_half2(((ars * brs) + (ais * bis)) * oos,
                    ((ais * brs) - (ars * bis)) * oos);
  return quot;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Otherwise we would lose half the exponent range. There are
 * various ways of doing guarded computation. For now chose the simplest
 * and fastest solution, however this may suffer from inaccuracies if hsqrt
 * and division are not IEEE compliant.
 */
__device__ static __inline__ half
abs_c_f16(half2 x)
{
  half a = real_c_f16(x);
  half b = imag_c_f16(x);
  half v, w, t;
  a = __habs(a);
  b = __habs(b);
  if (a > b) {
    v = a;
    w = b;
  } else {
    v = b;
    w = a;
  }
  t = w / v;
  t = (half)1.0 + t * t;
  t = v * hsqrt(t);
  if ((v == (half)0.0) || (v > (half)65504) || (w > (half)65504)) {
    t = v + w;
  }
  return t;
}

__device__ __host__ __forceinline__ cuFloatComplex
__half22complex(const half2& val)
{
  cuFloatComplex val_complex = {};
  val_complex.x = __half2float(val.x);
  val_complex.y = __half2float(val.y);
  return val_complex;
}

__device__ __forceinline__ half2
operator+(const half2& lh, const half2& rh)
{
  return add_c_f16(lh, rh);
}
__device__ __forceinline__ half2
operator-(const half2& lh, const half2& rh)
{
  return sub_c_f16(lh, rh);
}
__device__ __forceinline__ half2
operator*(const half2& lh, const half2& rh)
{
  return mul_c_f16(lh, rh);
}
__device__ __forceinline__ half2
operator/(const half2& lh, const half2& rh)
{
  return div_c_f16(lh, rh);
}

__device__ __forceinline__ half2&
operator+=(half2& lh, const half2& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ half2&
operator-=(half2& lh, const half2& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ half2&
operator*=(half2& lh, const half2& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ half2&
operator/=(half2& lh, const half2& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ half2
operator+(const half2& lh, const cuFloatComplex& rh)
{
  return lh + make_half2(rh.x, rh.y);
}
__device__ __forceinline__ half2
operator-(const half2& lh, const cuFloatComplex& rh)
{
  return lh - make_half2(rh.x, rh.y);
}
__device__ __forceinline__ half2
operator*(const half2& lh, const cuFloatComplex& rh)
{
  return lh * make_half2(rh.x, rh.y);
}
__device__ __forceinline__ half2
operator/(const half2& lh, const cuFloatComplex& rh)
{
  return lh / make_half2(rh.x, rh.y);
}

__device__ __forceinline__ half2
operator+(const cuFloatComplex& lh, const half2& rh)
{
  return rh + make_half2(lh.x, lh.y);
}
__device__ __forceinline__ half2
operator-(const cuFloatComplex& lh, const half2& rh)
{
  return rh - make_half2(lh.x, lh.y);
}
__device__ __forceinline__ half2
operator*(const cuFloatComplex& lh, const half2& rh)
{
  return rh * make_half2(lh.x, lh.y);
}
__device__ __forceinline__ half2
operator/(const cuFloatComplex& lh, const half2& rh)
{
  return rh / make_half2(lh.x, lh.y);
}

__device__ __forceinline__ half2&
operator+=(half2& lh, const cuFloatComplex& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ half2&
operator-=(half2& lh, const cuFloatComplex& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ half2&
operator*=(half2& lh, const cuFloatComplex& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ half2&
operator/=(half2& lh, const cuFloatComplex& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ cuFloatComplex&
operator+=(cuFloatComplex& lh, const half2& rh)
{
  lh = lh + __half22complex(rh);
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator-=(cuFloatComplex& lh, const half2& rh)
{
  lh = lh - __half22complex(rh);
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator*=(cuFloatComplex& lh, const half2& rh)
{
  lh = lh * __half22complex(rh);
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator/=(cuFloatComplex& lh, const half2& rh)
{
  lh = lh / __half22complex(rh);
  return lh;
}

__device__ __forceinline__ half2
conj(const half2& z)
{
  return make_half2(real_c_f16(z), -imag_c_f16(z));
}

__device__ __forceinline__ bool
operator==(const half2& lh, const half2& rh)
{
  return lh.x == rh.x && lh.y == rh.y;
}

__device__ __forceinline__ bool
operator!=(const half2& lh, const half2& rh)
{
  return lh.x != rh.x || lh.y != rh.y;
}

__device__ __forceinline__ half2
operator-(const half2& z)
{
  return make_half2(-z.x, -z.y);
}

// __device__ __forceinline__ half2
// atomicAdd(half2* lh, const cuFloatComplex& rh)
// {
//   atomicAdd(&(lh->x), rh.x);
//   atomicAdd(&(lh->y), rh.y);
//   return *lh;
// }

__host__ __device__ __forceinline__ bool
is_zero(const half2& z)
{
  return __half2float(z.x) == 0.f && __half2float(z.y) == 0.f;
}

__device__ __forceinline__ half
alpha_abs(const half2& z)
{
  half real = alpha_abs(z.x);
  half imag = alpha_abs(z.y);
  if (real > imag) {
    imag /= real;
    return real * alpha_sqrt(imag * imag + 1);
  } else if (imag != half{}) {
    real /= imag;
    return imag * alpha_sqrt(real * real + 1);
  }
  return 0.f;
}

__device__ __forceinline__ half2
alpha_sqrt(const half2& z)
{
  half x = z.x;
  half y = z.y;

  half sgnp;
  if (y < half{}) {
    sgnp = -1.0f;
  } else {
    sgnp = 1.0f;
  }
  half absz = alpha_abs(z);

  return make_half2(hsqrt((absz + x) * 0.5f), sgnp * hsqrt((absz - x) * 0.5f));
  // return z;
}

__device__ __forceinline__ half2
alpha_fma(half2 p, half2 q, half2 r)
{
  half real = alpha_fma(-p.y, q.y, alpha_fma(p.x, q.x, r.x));
  half imag = alpha_fma(p.x, q.y, alpha_fma(p.y, q.x, r.y));
  return make_half2(real, imag);
}

#endif /* C_F16_TYPES_H */
