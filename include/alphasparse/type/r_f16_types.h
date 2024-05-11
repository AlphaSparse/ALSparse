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



#ifndef R_F16_TYPES_H
#define R_F16_TYPES_H

#include <cuda_fp16.h>

/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ half
operator+(const half& lh, const float& rh)
{
  return lh + __float2half(rh);
}
__device__ __forceinline__ half
operator-(const half& lh, const float& rh)
{
  return lh - __float2half(rh);
}
__device__ __forceinline__ half
operator*(const half& lh, const float& rh)
{
  return lh * __float2half(rh);
}
__device__ __forceinline__ half
operator/(const half& lh, const float& rh)
{
  return lh / __float2half(rh);
}

__device__ __forceinline__ half&
operator+=(half& lh, const float& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ half&
operator-=(half& lh, const float& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ half&
operator*=(half& lh, const float& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ half&
operator/=(half& lh, const float& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ half
operator+(const float& lh, const half& rh)
{
  return __float2half(lh) + rh;
}
__device__ __forceinline__ half
operator-(const float& lh, const half& rh)
{
  return __float2half(lh) - rh;
}
__device__ __forceinline__ half
operator*(const float& lh, const half& rh)
{
  return __float2half(lh) * rh;
}
__device__ __forceinline__ half
operator/(const float& lh, const half& rh)
{
  return __float2half(lh) / rh;
}

__device__ __forceinline__ float&
operator+=(float& lh, const half& rh)
{
  lh = lh + __half2float(rh);
  return lh;
}
__device__ __forceinline__ float&
operator-=(float& lh, const half& rh)
{
  lh = lh - __half2float(rh);
  return lh;
}
__device__ __forceinline__ float&
operator*=(float& lh, const half& rh)
{
  lh = lh * __half2float(rh);
  return lh;
}
__device__ __forceinline__ float&
operator/=(float& lh, const half& rh)
{
  lh = lh / __half2float(rh);
  return lh;
}

__device__ __forceinline__ half
conj(const half& z)
{
  return z;
}

__host__ __device__ __forceinline__ bool
is_zero(const half& z)
{
  return __half2float(z) == 0.f;
}

// __host__ __device__ __forceinline__ bool
// operator==(const half& lh, const half& rh)
// {
//   return __heq(lh, rh);
// }

// __host__ __device__ __forceinline__ bool
// operator!=(const half& lh, const half& rh)
// {
//   return __hneu(lh, rh);
// }

__device__ __forceinline__ half
alpha_abs(const half& z)
{
  return z;
}

__device__ __forceinline__ half
alpha_sqrt(const half& z)
{
  return hsqrt(z);
}

__device__ __forceinline__ half alpha_fma(half p, half q, half r){
  return __hfma(p, q, r);
}

#endif /* R_F16_TYPES_H */
