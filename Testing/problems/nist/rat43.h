#pragma once

#include <cmath>

#include "api/dispatch.h"

// Rat43 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Rat43.dat

namespace solverslib::test
{
inline const double* rat43_y()
{
    static const double y[15] = {
    16.08, 33.83, 65.8, 97.2, 191.55, 326.2, 386.87, 520.53, 590.03, 651.92, 724.93, 699.56, 
    689.96, 637.56, 717.41
    };
    return y;
}

inline const double* rat43_x()
{
    static const double x[15] = {
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
    };
    return x;
}

struct rat43_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        using std::pow;
        const double* y = rat43_y();
        const double* x_data = rat43_x();
        for (int i = 0; i < 15; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0] / pow(T(1) + exp(b[1]-b[2]*x), T(1)/b[3]));
        }
    }
};

inline void rat43_residuals(const vector_type& b, vector_type& r)
{
    r.resize(15);
    rat43_model{}(b.data(), r.data());
}

inline api::least_squares_problem rat43_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 4;
    problem.num_residuals  = 15;
    problem.residuals      = rat43_residuals;
    return problem;
}

inline vector_type rat43_start1()
{
    vector_type b(4);
    b << 100.0, 10.0, 1.0, 1.0;
    return b;
}

inline vector_type rat43_start2()
{
    vector_type b(4);
    b << 700.0, 5.0, 0.75, 1.3;
    return b;
}

inline constexpr double rat43_certified_rss = 8786.404908;

inline vector_type rat43_certified_parameters()
{
    vector_type b(4);
    b << 699.6415127, 5.2771253025, 0.75962938329, 1.2792483859;
    return b;
}
}  // namespace solverslib::test
