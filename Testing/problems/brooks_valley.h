#pragma once

#include <cmath>

#include "api/dispatch.h"

// Brooks's coupled curved valley (Brooks 2024, "Higher-Order Corrections to
// Optimisers based on Newton's Method", v2, Sections 6-7).
//   r = [x + y^2, K*(y - x^2)]
// Start (pi, e). K is swept across {1, 10, ..., 1e12} in the literature.
// Minimum at (0,0), F(x*) = 0, distinct from both Rosenbrock models: here
// both residuals are individually quadratic, coupled through K's scale
// rather than through a single polynomial valley.

namespace solverslib::test
{
struct brooks_valley_model
{
    double k = 1.0;

    template <class T> void operator()(const T* x, T* r) const
    {
        r[0] = x[0] + x[1] * x[1];
        r[1] = T(k) * (x[1] - x[0] * x[0]);
    }
};

inline vector_type brooks_valley_start()
{
    vector_type x(2);
    x << 3.14159265358979323846, 2.71828182845904523536;  // pi, e
    return x;
}

inline api::least_squares_problem brooks_valley_least_squares(double k)
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 2;
    problem.residuals      = [k](const vector_type& x, vector_type& r)
    {
        r.resize(2);
        brooks_valley_model{k}(x.data(), r.data());
    };
    return problem;
}

inline constexpr double brooks_valley_certified_rss = 0.0;
}  // namespace solverslib::test
