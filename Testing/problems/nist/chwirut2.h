#pragma once

#include <cmath>

#include "api/dispatch.h"

// Chwirut2 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Chwirut2.dat

namespace solverslib::test
{
inline const double* chwirut2_y()
{
    static const double y[54] = {
    92.9, 57.1, 31.05, 11.5875, 8.025, 63.6, 21.4, 14.25, 8.475, 63.8, 26.8, 16.4625, 7.125, 
    67.3, 41.0, 21.15, 8.175, 81.5, 13.12, 59.9, 14.62, 32.9, 5.44, 12.56, 5.44, 32.0, 13.95, 
    75.8, 20.0, 10.42, 59.5, 21.67, 8.55, 62.0, 20.2, 7.76, 3.75, 11.81, 54.7, 23.7, 11.55, 
    61.3, 17.7, 8.74, 59.2, 16.3, 8.62, 81.0, 4.87, 14.62, 81.7, 17.17, 81.3, 28.9
    };
    return y;
}

inline const double* chwirut2_x()
{
    static const double x[54] = {
    0.5, 1.0, 1.75, 3.75, 5.75, 0.875, 2.25, 3.25, 5.25, 0.75, 1.75, 2.75, 4.75, 0.625, 1.25, 
    2.25, 4.25, 0.5, 3.0, 0.75, 3.0, 1.5, 6.0, 3.0, 6.0, 1.5, 3.0, 0.5, 2.0, 4.0, 0.75, 2.0, 
    5.0, 0.75, 2.25, 3.75, 5.75, 3.0, 0.75, 2.5, 4.0, 0.75, 2.5, 4.0, 0.75, 2.5, 4.0, 0.5, 6.0, 
    3.0, 0.5, 2.75, 0.5, 1.75
    };
    return x;
}

struct chwirut2_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = chwirut2_y();
        const double* x_data = chwirut2_x();
        for (int i = 0; i < 54; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (exp(-b[0]*x) / (b[1] + b[2]*x));
        }
    }
};

inline void chwirut2_residuals(const vector_type& b, vector_type& r)
{
    r.resize(54);
    chwirut2_model{}(b.data(), r.data());
}

inline api::least_squares_problem chwirut2_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 3;
    problem.num_residuals  = 54;
    problem.residuals      = chwirut2_residuals;
    return problem;
}

inline vector_type chwirut2_start1()
{
    vector_type b(3);
    b << 0.1, 0.01, 0.02;
    return b;
}

inline vector_type chwirut2_start2()
{
    vector_type b(3);
    b << 0.15, 0.008, 0.01;
    return b;
}

inline constexpr double chwirut2_certified_rss = 513.04802941;

inline vector_type chwirut2_certified_parameters()
{
    vector_type b(3);
    b << 0.16657666537, 0.0051653291286, 0.012150007096;
    return b;
}
}  // namespace solverslib::test
