#pragma once

#include <cmath>

#include "api/dispatch.h"

// Misra1c (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1c.dat

namespace solverslib::test
{
inline const double* misra1c_y()
{
    static const double y[14] = {
    10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.4, 75.47, 
    81.78
    };
    return y;
}

inline const double* misra1c_x()
{
    static const double x[14] = {
    77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 
    760.0
    };
    return x;
}

struct misra1c_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::pow;
        const double* y = misra1c_y();
        const double* x_data = misra1c_x();
        for (int i = 0; i < 14; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0]*(T(1) - pow(T(1) + T(2)*b[1]*x, -0.5)));
        }
    }
};

inline void misra1c_residuals(const vector_type& b, vector_type& r)
{
    r.resize(14);
    misra1c_model{}(b.data(), r.data());
}

inline api::least_squares_problem misra1c_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 14;
    problem.residuals      = misra1c_residuals;
    return problem;
}

inline vector_type misra1c_start1()
{
    vector_type b(2);
    b << 500.0, 0.0001;
    return b;
}

inline vector_type misra1c_start2()
{
    vector_type b(2);
    b << 600.0, 0.0002;
    return b;
}

inline constexpr double misra1c_certified_rss = 0.040966836971;

inline vector_type misra1c_certified_parameters()
{
    vector_type b(2);
    b << 636.42725809, 0.00020813627256;
    return b;
}
}  // namespace solverslib::test
