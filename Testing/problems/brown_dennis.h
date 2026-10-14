#pragma once

#include <cmath>
#include <cstddef>

#include "api/dispatch.h"

// Brown and Dennis function (More-Garbow-Hillstrom #16; Moré 1978 report, #4).
//   t_i = i/5,  i = 1..m
//   r_i(x) = (x1 + t_i*x2 - exp(t_i))^2 + (x3 + x4*sin(t_i) - cos(t_i))^2
// Start x0 = (25, 5, -5, -1). Large nonzero-residual minimum: for m = 20,
// the published minimum sum of squares is F(x*) ~= 85822.2.

namespace solverslib::test
{
inline vector_type brown_dennis_start()
{
    vector_type x(4);
    x << 25.0, 5.0, -5.0, -1.0;
    return x;
}

template <std::size_t M = 20> struct brown_dennis_model
{
    template <class T> void operator()(const T* x, T* r) const
    {
        using std::cos;
        using std::exp;
        using std::sin;
        for (std::size_t i = 1; i <= M; ++i)
        {
            const double t  = static_cast<double>(i) / 5.0;
            const T      a  = x[0] + T(t) * x[1] - T(exp(t));
            const T      b  = x[2] + x[3] * T(sin(t)) - T(cos(t));
            r[i - 1]        = a * a + b * b;
        }
    }
};

inline void brown_dennis_residuals(const vector_type& x, vector_type& r)
{
    r.resize(20);
    brown_dennis_model<20>{}(x.data(), r.data());
}

inline api::least_squares_problem brown_dennis_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 4;
    problem.num_residuals  = 20;
    problem.residuals      = brown_dennis_residuals;
    return problem;
}

inline constexpr double brown_dennis_certified_rss = 85822.20162635137;
}  // namespace solverslib::test
