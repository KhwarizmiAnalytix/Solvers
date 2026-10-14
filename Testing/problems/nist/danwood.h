#pragma once

#include <cmath>

#include "api/dispatch.h"

// DanWood (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/DanWood.dat

namespace solverslib::test
{
inline const double* danwood_y()
{
    static const double y[6] = {
    2.138, 3.421, 3.597, 4.34, 4.882, 5.66
    };
    return y;
}

inline const double* danwood_x()
{
    static const double x[6] = {
    1.309, 1.471, 1.49, 1.565, 1.611, 1.68
    };
    return x;
}

struct danwood_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::pow;
        const double* y = danwood_y();
        const double* x_data = danwood_x();
        for (int i = 0; i < 6; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0]*pow(x, b[1]));
        }
    }
};

inline void danwood_residuals(const vector_type& b, vector_type& r)
{
    r.resize(6);
    danwood_model{}(b.data(), r.data());
}

inline api::least_squares_problem danwood_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 6;
    problem.residuals      = danwood_residuals;
    return problem;
}

inline vector_type danwood_start1()
{
    vector_type b(2);
    b << 1.0, 5.0;
    return b;
}

inline vector_type danwood_start2()
{
    vector_type b(2);
    b << 0.7, 4.0;
    return b;
}

inline constexpr double danwood_certified_rss = 0.0043173084083;

inline vector_type danwood_certified_parameters()
{
    vector_type b(2);
    b << 0.76886226176, 3.8604055871;
    return b;
}
}  // namespace solverslib::test
