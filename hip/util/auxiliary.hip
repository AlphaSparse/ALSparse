#include <sstream>
#include <hip/hip_complex.h>
#include "alphasparse/util/auxiliary.h"

std::ostream& operator<<(std::ostream& out, const hipFloatComplex& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

std::ostream& operator<<(std::ostream& out, const hipDoubleComplex& z)
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
