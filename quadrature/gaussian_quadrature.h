#pragma once

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits

#include "MathModule.h"     // for MATH_API
#include "common/macros.h"  // for QUARISMA_DELETE_CLASS

namespace quarisma
{
template <typename value_t>
class vector;
}

namespace quarisma
{
class gaussian_quadrature
{
    QUARISMA_DELETE_CLASS(gaussian_quadrature);

public:
    MATH_API static void gauss_hermite_coefficients(
        size_t          np,
        vector<double>& points,
        vector<double>& weights,
        bool            use_precomputed = true,
        double          tolerance       = std::numeric_limits<double>::epsilon(),
        size_t          max_iter        = 20);

    MATH_API static void gauss_laguerre_coefficients(
        double          alpha,
        size_t          n,
        vector<double>& points,
        vector<double>& weights,
        double          tolerance = std::numeric_limits<double>::epsilon(),
        size_t          max_iter  = 20);

    MATH_API static void gauss_legendre_coefficients(
        size_t n, vector<double>& points, vector<double>& weights, double tolerance = 1.e-14);

    MATH_API static void gauss_kronrod(
        size_t          n,
        vector<double>& x,
        vector<double>& w1,
        vector<double>& w2,
        double          tolerance = 1.e-14,
        size_t          max_iter  = 20);
};
}  // namespace quarisma
