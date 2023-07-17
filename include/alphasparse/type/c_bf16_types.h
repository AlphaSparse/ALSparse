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



#ifndef C_BF16_TYPES_H
#define C_BF16_TYPES_H

// #define __PRINT_MACRO(x) #x
// #define PRINT_MARCO(x) #x"=" __PRINT_MACRO(x)
// #pragma message(PRINT_MARCO(__CUDA_ARCH__))

#include "c_f32_types.h"
#include <cuComplex.h>
#include <cuda_bf16.h>

__device__ __host__ __forceinline__ cuFloatComplex
__bfloat1622complex(const nv_bfloat162& val)
{
  cuFloatComplex val_complex = {};
  val_complex.x = __bfloat162float(val.x);
  val_complex.y = __bfloat162float(val.y);
  return val_complex;
}

#if (CUDA_ARCH >= 80)
__host__ __device__ static __inline__ nv_bfloat16
real_c_bf16(nv_bfloat162 x)
{
  return x.x;
}

__host__ __device__ static __inline__ nv_bfloat16
imag_c_bf16(nv_bfloat162 x)
{
  return x.y;
}

__device__ static __inline__ nv_bfloat162
conj_c_bf16(nv_bfloat162 x)
{
  return make_bfloat162(real_c_bf16(x), -x.y);
}

__device__ static __inline__ nv_bfloat162
add_c_bf16(nv_bfloat162 x, nv_bfloat162 y)
{
  return make_bfloat162(real_c_bf16(x) + real_c_bf16(y),
                        imag_c_bf16(x) + imag_c_bf16(y));
}

__device__ static __inline__ nv_bfloat162
sub_c_bf16(nv_bfloat162 x, nv_bfloat162 y)
{
  return make_bfloat162(real_c_bf16(x) - real_c_bf16(y),
                        imag_c_bf16(x) - imag_c_bf16(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we
 * don't do it either to stay competitive.
 */
__device__ static __inline__ nv_bfloat162
mul_c_bf16(nv_bfloat162 x, nv_bfloat162 y)
{
  nv_bfloat162 prod;
  prod = make_bfloat162(
    (real_c_bf16(x) * real_c_bf16(y)) - (imag_c_bf16(x) * imag_c_bf16(y)),
    (real_c_bf16(x) * imag_c_bf16(y)) + (imag_c_bf16(x) * real_c_bf16(y)));
  return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
__device__ static __inline__ nv_bfloat162
div_c_bf16(nv_bfloat162 x, nv_bfloat162 y)
{
  nv_bfloat162 quot;
  nv_bfloat16 s = (__habs(real_c_bf16(y))) + (__habs(imag_c_bf16(y)));
  nv_bfloat16 oos = (nv_bfloat16)1.0 / s;
  nv_bfloat16 ars = real_c_bf16(x) * oos;
  nv_bfloat16 ais = imag_c_bf16(x) * oos;
  nv_bfloat16 brs = real_c_bf16(y) * oos;
  nv_bfloat16 bis = imag_c_bf16(y) * oos;
  s = (brs * brs) + (bis * bis);
  oos = (nv_bfloat16)1.0 / s;
  quot = make_bfloat162(((ars * brs) + (ais * bis)) * oos,
                        ((ais * brs) - (ars * bis)) * oos);
  return quot;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Otherwise we would lose nv_bfloat16 the exponent range. There are
 * various ways of doing guarded computation. For now chose the simplest
 * and fastest solution, however this may suffer from inaccuracies if hsqrt
 * and division are not IEEE compliant.
 */
__device__ static __inline__ nv_bfloat16
abs_c_bf16(nv_bfloat162 x)
{
  nv_bfloat16 a = real_c_bf16(x);
  nv_bfloat16 b = imag_c_bf16(x);
  nv_bfloat16 v, w, t;
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
  t = (nv_bfloat16)1.0 + t * t;
  t = v * hsqrt(t);
  if ((v == (nv_bfloat16)0.0) || (v > (nv_bfloat16)65504) ||
      (w > (nv_bfloat16)65504)) {
    t = v + w;
  }
  return t;
}

__device__ __forceinline__ nv_bfloat162
operator+(const nv_bfloat162& lh, const nv_bfloat162& rh)
{
  return add_c_bf16(lh, rh);
}
__device__ __forceinline__ nv_bfloat162
operator-(const nv_bfloat162& lh, const nv_bfloat162& rh)
{
  return sub_c_bf16(lh, rh);
}
__device__ __forceinline__ nv_bfloat162
operator*(const nv_bfloat162& lh, const nv_bfloat162& rh)
{
  return mul_c_bf16(lh, rh);
}
__device__ __forceinline__ nv_bfloat162
operator/(const nv_bfloat162& lh, const nv_bfloat162& rh)
{
  return div_c_bf16(lh, rh);
}

__device__ __forceinline__ nv_bfloat162&
operator+=(nv_bfloat162& lh, const nv_bfloat162& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat162&
operator-=(nv_bfloat162& lh, const nv_bfloat162& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat162&
operator*=(nv_bfloat162& lh, const nv_bfloat162& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat162&
operator/=(nv_bfloat162& lh, const nv_bfloat162& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ nv_bfloat162
operator+(const nv_bfloat162& lh, const cuFloatComplex& rh)
{
  return lh + make_bfloat162(rh.x, rh.y);
}
__device__ __forceinline__ nv_bfloat162
operator-(const nv_bfloat162& lh, const cuFloatComplex& rh)
{
  return lh - make_bfloat162(rh.x, rh.y);
}
__device__ __forceinline__ nv_bfloat162
operator*(const nv_bfloat162& lh, const cuFloatComplex& rh)
{
  return lh * make_bfloat162(rh.x, rh.y);
}
__device__ __forceinline__ nv_bfloat162
operator/(const nv_bfloat162& lh, const cuFloatComplex& rh)
{
  return lh / make_bfloat162(rh.x, rh.y);
}

__device__ __forceinline__ nv_bfloat162
operator+(const cuFloatComplex& lh, const nv_bfloat162& rh)
{
  return rh + make_bfloat162(lh.x, lh.y);
}
__device__ __forceinline__ nv_bfloat162
operator-(const cuFloatComplex& lh, const nv_bfloat162& rh)
{
  return rh - make_bfloat162(lh.x, lh.y);
}
__device__ __forceinline__ nv_bfloat162
operator*(const cuFloatComplex& lh, const nv_bfloat162& rh)
{
  return rh * make_bfloat162(lh.x, lh.y);
}
__device__ __forceinline__ nv_bfloat162
operator/(const cuFloatComplex& lh, const nv_bfloat162& rh)
{
  return rh / make_bfloat162(lh.x, lh.y);
}

__device__ __forceinline__ nv_bfloat162&
operator+=(nv_bfloat162& lh, const cuFloatComplex& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat162&
operator-=(nv_bfloat162& lh, const cuFloatComplex& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat162&
operator*=(nv_bfloat162& lh, const cuFloatComplex& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat162&
operator/=(nv_bfloat162& lh, const cuFloatComplex& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ cuFloatComplex&
operator+=(cuFloatComplex& lh, const nv_bfloat162& rh)
{
  lh = lh + __bfloat1622complex(rh);
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator-=(cuFloatComplex& lh, const nv_bfloat162& rh)
{
  lh = lh - __bfloat1622complex(rh);
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator*=(cuFloatComplex& lh, const nv_bfloat162& rh)
{
  lh = lh * __bfloat1622complex(rh);
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator/=(cuFloatComplex& lh, const nv_bfloat162& rh)
{
  lh = lh / __bfloat1622complex(rh);
  return lh;
}

__device__ __forceinline__ nv_bfloat162
conj(const nv_bfloat162& z)
{
  return make_bfloat162(real_c_bf16(z), -imag_c_bf16(z));
}

__device__ __forceinline__ bool
operator==(const nv_bfloat162& lh, const nv_bfloat162& rh)
{
  return lh.x == rh.x && lh.y == rh.y;
}

__device__ __forceinline__ bool
operator!=(const nv_bfloat162& lh, const nv_bfloat162& rh)
{
  return lh.x != rh.x || lh.y != rh.y;
}

__device__ __forceinline__ nv_bfloat162
operator-(const nv_bfloat162& z)
{
  return make_bfloat162(-z.x, -z.y);
}

__host__ __device__ __forceinline__ bool
is_zero(const nv_bfloat162& z)
{
  return __bfloat162float(z.x) == 0.f && __bfloat162float(z.y) == 0.f;
}

__device__ __forceinline__ nv_bfloat162
// alpha_sqrt(const nv_bfloat162& z)
// {
//   return z;
// }

__device__ __forceinline__ nv_bfloat16
alpha_abs(const nv_bfloat162& z)
{
  nv_bfloat16 real = alpha_abs(z.x);
  nv_bfloat16 imag = alpha_abs(z.y);
  if (real > imag) {
    imag /= real;
    return real * alpha_sqrt(imag * imag + 1);
  } else if (imag != nv_bfloat16{}) {
    real /= imag;
    return imag * alpha_sqrt(real * real + 1);
  }
  return 0.f;
}

__device__ __forceinline__ nv_bfloat162
alpha_sqrt(const nv_bfloat162& z)
{
  nv_bfloat16 x = z.x;
  nv_bfloat16 y = z.y;

  nv_bfloat16 sgnp;
  if(y < nv_bfloat16{}){
    sgnp = -1.0f;
  }
  else{
    sgnp = 1.0f;
  }
  nv_bfloat16 absz = alpha_abs(z);

  return make_bfloat162(hsqrt((absz + x) * 0.5f), sgnp * hsqrt((absz - x) * 0.5f));
  // return z;
}

__device__ __forceinline__ nv_bfloat162 alpha_fma(nv_bfloat162 p, nv_bfloat162 q, nv_bfloat162 r){
  nv_bfloat16 real = alpha_fma(-p.y, q.y, alpha_fma(p.x, q.x, r.x));
  nv_bfloat16 imag = alpha_fma(p.x, q.y, alpha_fma(p.y, q.x, r.y));
  return make_bfloat162(real, imag);
}

#endif
#endif /* C_BF16_TYPES_H */
