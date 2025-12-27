#pragma once

#include "MathModule.h"
#include "common/macros.h"

namespace quarisma
{
class inverse_gaussian_distribution
{
    QUARISMA_DELETE_CLASS(inverse_gaussian_distribution);

public:
    MATH_API static double density(double x, double lambda);

    MATH_API static double cdf(double x, double lambda);
};
}  // namespace quarisma
