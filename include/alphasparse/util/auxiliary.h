#pragma once

#include <sstream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>

std::ostream& operator<<(std::ostream& out, const cuFloatComplex& z);

std::ostream& operator<<(std::ostream& out, const cuDoubleComplex& z);

std::ostream& operator<<(std::ostream& out, const half& z);

std::ostream& operator<<(std::ostream& out, const half2& z);

std::ostream& operator<<(std::ostream& out, const nv_bfloat16& z);

std::ostream& operator<<(std::ostream& out, const nv_bfloat162& z);

std::ostream& operator<<(std::ostream& out, const int8_t& z);
