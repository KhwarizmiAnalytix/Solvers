#pragma once

#include <cstddef>

#include "MathModule.h"
#include "common/macros.h"

namespace quarisma
{
class normal_distribution
{
    QUARISMA_DELETE_CLASS(normal_distribution)

public:
    /**
    //tex:
    // $$n(x)=\frac{\exp{(-\frac{z^2}{2})}}{\sqrt{2\pi}}$$
    **/
    MATH_API static double density(double x);

    /**
    //tex:
    //$$N(x)=\int_{-\infty}^{x}\frac{\exp{(-\frac{u^2}{2})}}{\sqrt{2\pi}}du$$
    **/
    MATH_API static double cdf(double z);

    /**
    //tex:
    //$$\frac{N(x)}{n(x)}$$
    **/
    MATH_API static double cdf_over_density(double z);

    /**
    //tex:
    //$$N^{-1}(x)$$
    **/
    MATH_API static double inv_cdf(double p);

    MATH_API static double inv_cdf_fast(double p);

    MATH_API static void inv_cdf(size_t size, const double* uniform, double* gaussian);
};
}  // namespace quarisma
