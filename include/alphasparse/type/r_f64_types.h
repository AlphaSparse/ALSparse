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



#ifndef R_F64_TYPES_H
#define R_F64_TYPES_H

__device__ __forceinline__ double
conj(const double& z)
{
  return z;
}

__host__ __device__ __forceinline__ bool
is_zero(const double& z)
{
  return z == 0.f;
}

__device__ __forceinline__ double
alpha_abs(const double& z)
{
  return fabs(z);
}

__device__ __forceinline__ double
alpha_sqrt(const double& z)
{
  return sqrt(z);
}

template <typename T>
__device__ __forceinline__ T
alpha_cast(const double& z)
{
  return z;
}

__device__ __forceinline__ bool alpha_gt(const double& x, const double& y) { return x > y; }

__device__ __forceinline__ double alpha_fma(double p, double q, double r){
  return std::fma(p, q, r);
}

#endif /* R_F64_TYPES_H */
