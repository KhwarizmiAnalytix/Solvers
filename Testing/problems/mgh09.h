#pragma once

#include "api/dispatch.h"

// Kowalik-Osborne / MGH09 (NIST StRD, "Higher" difficulty; More-Garbow-Hillstrom
// #2; Nielsen 1999, problem 9). Rational model:
//   y = b1*(u^2 + u*b2) / (u^2 + u*b3 + b4)
// Data and certified values from NIST's StRD nonlinear-regression archive
// (https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/MGH09.dat),
// cross-checked against the vendored copy at
// ThirdParty/ceres/data/nist/MGH09.dat.

namespace solverslib::test
{
inline const double* mgh09_u()
{
    static const double u[11] = {
        4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625};
    return u;
}

inline const double* mgh09_y()
{
    static const double y[11] = {0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342,
        0.0323, 0.0235, 0.0246};
    return y;
}

struct mgh09_model
{
    template <class T> void operator()(const T* x, T* r) const
    {
        const double* u = mgh09_u();
        const double* y = mgh09_y();
        for (int i = 0; i < 11; ++i)
        {
            const T ui = T(u[i]);
            r[i]       = T(y[i]) - x[0] * (ui * ui + ui * x[1]) / (ui * ui + ui * x[2] + x[3]);
        }
    }
};

inline void mgh09_residuals(const vector_type& x, vector_type& r)
{
    r.resize(11);
    mgh09_model{}(x.data(), r.data());
}

inline api::least_squares_problem mgh09_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 4;
    problem.num_residuals  = 11;
    problem.residuals      = mgh09_residuals;
    return problem;
}

inline vector_type mgh09_start1()
{
    vector_type x(4);
    x << 25.0, 39.0, 41.5, 39.0;
    return x;
}

inline vector_type mgh09_start2()
{
    vector_type x(4);
    x << 0.25, 0.39, 0.415, 0.39;
    return x;
}

inline constexpr double mgh09_certified_rss = 3.0750560385e-04;

inline vector_type mgh09_certified_parameters()
{
    vector_type x(4);
    x << 1.9280693458e-01, 1.9128232873e-01, 1.2305650693e-01, 1.3606233068e-01;
    return x;
}
}  // namespace solverslib::test
