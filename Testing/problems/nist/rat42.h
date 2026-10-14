#pragma once

#include <cmath>

#include "api/dispatch.h"

// Rat42 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Rat42.dat

namespace solverslib::test
{
inline const double* rat42_y()
{
    static const double y[9] = {
    8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08
    };
    return y;
}

inline const double* rat42_x()
{
    static const double x[9] = {
    9.0, 14.0, 21.0, 28.0, 42.0, 57.0, 63.0, 70.0, 79.0
    };
    return x;
}

struct rat42_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = rat42_y();
        const double* x_data = rat42_x();
        for (int i = 0; i < 9; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0] / (T(1) + exp(b[1]-b[2]*x)));
        }
    }
};

inline void rat42_residuals(const vector_type& b, vector_type& r)
{
    r.resize(9);
    rat42_model{}(b.data(), r.data());
}

inline api::least_squares_problem rat42_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 3;
    problem.num_residuals  = 9;
    problem.residuals      = rat42_residuals;
    return problem;
}

inline vector_type rat42_start1()
{
    vector_type b(3);
    b << 100.0, 1.0, 0.1;
    return b;
}

inline vector_type rat42_start2()
{
    vector_type b(3);
    b << 75.0, 2.5, 0.07;
    return b;
}

inline constexpr double rat42_certified_rss = 8.0565229338;

inline vector_type rat42_certified_parameters()
{
    vector_type b(3);
    b << 72.462237576, 2.6180768402, 0.067359200066;
    return b;
}
}  // namespace solverslib::test
