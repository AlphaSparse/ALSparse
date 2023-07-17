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



#ifndef R_BF16_TYPES_H
#define R_BF16_TYPES_H

#include <cuda_bf16.h>
#if (CUDA_ARCH >= 80)
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ nv_bfloat16
operator+(const nv_bfloat16& lh, const float& rh)
{
  return lh + __float2bfloat16(rh);
}
__device__ __forceinline__ nv_bfloat16
operator-(const nv_bfloat16& lh, const float& rh)
{
  return lh - __float2bfloat16(rh);
}
__device__ __forceinline__ nv_bfloat16
operator*(const nv_bfloat16& lh, const float& rh)
{
  return lh * __float2bfloat16(rh);
}
__device__ __forceinline__ nv_bfloat16
operator/(const nv_bfloat16& lh, const float& rh)
{
  return lh / __float2bfloat16(rh);
}

__device__ __forceinline__ nv_bfloat16&
operator+=(nv_bfloat16& lh, const float& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat16&
operator-=(nv_bfloat16& lh, const float& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat16&
operator*=(nv_bfloat16& lh, const float& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ nv_bfloat16&
operator/=(nv_bfloat16& lh, const float& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ nv_bfloat16
operator+(const float& lh, const nv_bfloat16& rh)
{
  return __float2bfloat16(lh) + rh;
}
__device__ __forceinline__ nv_bfloat16
operator-(const float& lh, const nv_bfloat16& rh)
{
  return __float2bfloat16(lh) - rh;
}
__device__ __forceinline__ nv_bfloat16
operator*(const float& lh, const nv_bfloat16& rh)
{
  return __float2bfloat16(lh) * rh;
}
__device__ __forceinline__ nv_bfloat16
operator/(const float& lh, const nv_bfloat16& rh)
{
  return __float2bfloat16(lh) / rh;
}

__device__ __forceinline__ float&
operator+=(float& lh, const nv_bfloat16& rh)
{
  lh = lh + __bfloat162float(rh);
  return lh;
}
__device__ __forceinline__ float&
operator-=(float& lh, const nv_bfloat16& rh)
{
  lh = lh - __bfloat162float(rh);
  return lh;
}
__device__ __forceinline__ float&
operator*=(float& lh, const nv_bfloat16& rh)
{
  lh = lh * __bfloat162float(rh);
  return lh;
}
__device__ __forceinline__ float&
operator/=(float& lh, const nv_bfloat16& rh)
{
  lh = lh / __bfloat162float(rh);
  return lh;
}

__device__ __forceinline__ nv_bfloat16
conj(const nv_bfloat16& z)
{
  return z;
}

__host__ __device__ __forceinline__ bool
is_zero(const nv_bfloat16& z)
{
  return __bfloat162float(z) == 0.f;
}

__device__ __forceinline__ nv_bfloat16
alpha_abs(const nv_bfloat16& z)
{
  return z;
}

__device__ __forceinline__ nv_bfloat16
alpha_sqrt(const nv_bfloat16& z)
{
  return hsqrt(z);
}

__device__ __forceinline__ nv_bfloat16 alpha_fma(nv_bfloat16 p, nv_bfloat16 q, nv_bfloat16 r){
  return __hfma(p, q, r);
}

#endif
#endif /* R_BF16_TYPES_H */
