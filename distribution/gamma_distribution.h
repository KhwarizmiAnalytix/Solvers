#pragma once

#include <complex>

#include "MathModule.h"
#include "common/macros.h"

namespace quarisma
{
class gamma_distribution
{
public:
    QUARISMA_DELETE_CLASS(gamma_distribution);

    MATH_API static double lgamma(double x);

    MATH_API static double gamma(double x);

    MATH_API static std::complex<double> lgamma(const std::complex<double>& x);
};
}  // namespace quarisma
