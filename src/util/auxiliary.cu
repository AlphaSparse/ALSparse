#include <sstream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>
#include "alphasparse/util/auxiliary.h"

std::ostream& operator<<(std::ostream& out, const cuFloatComplex& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const cuDoubleComplex& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const half& z)
{
    std::stringstream ss;
    ss << __half2float(z);
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const half2& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const nv_bfloat16& z)
{
    std::stringstream ss;
    ss << __bfloat162float(z);
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const nv_bfloat162& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const int8_t& z)
{
    std::stringstream ss;
    ss << int(z);
    return out << ss.str();
}
