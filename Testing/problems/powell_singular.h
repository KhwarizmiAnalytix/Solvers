#pragma once

#include <cmath>

#include "api/dispatch.h"

// Powell singular function (MINPACK TP13; Nielsen 1999, problem 6).
//   r1 = x1 + 10*x2
//   r2 = sqrt(5)*(x3 - x4)
//   r3 = (x2 - 2*x3)^2
//   r4 = sqrt(10)*(x1 - x4)^2
// Minimum at x* = (0,0,0,0), F(x*) = 0. The Jacobian is singular at the
// solution, which is the point of the test: it exercises damping behavior
// near a rank-deficient Jacobian rather than a generic smooth minimum.

namespace solverslib::test
{
inline vector_type powell_singular_start()
{
    vector_type x(4);
    x << 3.0, -1.0, 0.0, 1.0;
    return x;
}

struct powell_singular_model
{
    template <class T> void operator()(const T* x, T* r) const
    {
        r[0] = x[0] + T(10) * x[1];
        r[1] = T(std::sqrt(5.0)) * (x[2] - x[3]);
        r[2] = (x[1] - T(2) * x[2]) * (x[1] - T(2) * x[2]);
        r[3] = T(std::sqrt(10.0)) * (x[0] - x[3]) * (x[0] - x[3]);
    }
};

inline void powell_singular_residuals(const vector_type& x, vector_type& r)
{
    r.resize(4);
    powell_singular_model{}(x.data(), r.data());
}

inline api::least_squares_problem powell_singular_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 4;
    problem.num_residuals  = 4;
    problem.residuals      = powell_singular_residuals;
    return problem;
}

inline constexpr double powell_singular_certified_rss = 0.0;
}  // namespace solverslib::test
