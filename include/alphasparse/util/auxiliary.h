#pragma once

#include <sstream>
#ifdef __CUDA__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>
#endif
#ifdef __HIP__
#include <hip/hip_complex.h>
#endif

#ifdef __CUDA__
std::ostream& operator<<(std::ostream& out, const cuFloatComplex& z);

std::ostream& operator<<(std::ostream& out, const cuDoubleComplex& z);

std::ostream& operator<<(std::ostream& out, const half& z);

std::ostream& operator<<(std::ostream& out, const half2& z);

std::ostream& operator<<(std::ostream& out, const nv_bfloat16& z);

std::ostream& operator<<(std::ostream& out, const nv_bfloat162& z);
#endif

#ifdef __HIP__
std::ostream& operator<<(std::ostream& out, const hipFloatComplex& z);

std::ostream& operator<<(std::ostream& out, const hipDoubleComplex& z);
#endif

std::ostream& operator<<(std::ostream& out, const int8_t& z);




