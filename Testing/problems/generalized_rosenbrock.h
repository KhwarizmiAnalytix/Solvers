#pragma once

#include <cmath>

#include "api/dispatch.h"

// Generalized Rosenbrock valley (Liu-Zhang RNC-LM, Eq. 6 / Table 1a).
//   r = [x, A*(y - x^p/p)],  A = 1e6,  p in {2, 3}
// Start (1, 1/p). Minimum at (0,0), F(x*) = 0. The published success
// criterion is 0.5*||r||^2 < 1e-4 (not exact convergence), because the
// valley is numerically stiff for larger A.

namespace solverslib::test
{
template <int P> struct generalized_rosenbrock_model
{
    static constexpr double a = 1e6;

    template <class T> void operator()(const T* x, T* r) const
    {
        r[0] = x[0];
        T xp = x[0];
        for (int k = 1; k < P; ++k)
        {
            xp = xp * x[0];
        }
        r[1] = T(a) * (x[1] - xp / T(static_cast<double>(P)));
    }
};

template <int P> inline vector_type generalized_rosenbrock_start()
{
    vector_type x(2);
    x << 1.0, 1.0 / static_cast<double>(P);
    return x;
}

template <int P> inline void generalized_rosenbrock_residuals(const vector_type& x, vector_type& r)
{
    r.resize(2);
    generalized_rosenbrock_model<P>{}(x.data(), r.data());
}

template <int P> inline api::least_squares_problem generalized_rosenbrock_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 2;
    problem.residuals      = generalized_rosenbrock_residuals<P>;
    return problem;
}

inline constexpr double generalized_rosenbrock_certified_rss      = 0.0;
inline constexpr double generalized_rosenbrock_success_half_ssq   = 1e-4;
}  // namespace solverslib::test
