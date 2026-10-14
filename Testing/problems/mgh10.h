#pragma once

#include <cmath>

#include "api/dispatch.h"

// Meyer / MGH10 (NIST StRD, "Higher" difficulty; More-Garbow-Hillstrom problem
// referenced throughout the RNC-LM literature, e.g. LZ26 Table 1b). Exponential
// model:
//   y = b1 * exp(b2/(x+b3))
// Data and certified values from NIST's StRD nonlinear-regression archive
// (https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/MGH10.dat),
// cross-checked against the vendored copy at
// ThirdParty/ceres/data/nist/MGH10.dat.

namespace solverslib::test
{
inline const double* mgh10_x()
{
    static const double x[16] = {50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
        105.0, 110.0, 115.0, 120.0, 125.0};
    return x;
}

inline const double* mgh10_y()
{
    static const double y[16] = {34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0,
        9744.0, 8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0};
    return y;
}

struct mgh10_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* x = mgh10_x();
        const double* y = mgh10_y();
        for (int i = 0; i < 16; ++i)
        {
            r[i] = T(y[i]) - b[0] * exp(b[1] / (T(x[i]) + b[2]));
        }
    }
};

inline void mgh10_residuals(const vector_type& x, vector_type& r)
{
    r.resize(16);
    mgh10_model{}(x.data(), r.data());
}

inline api::least_squares_problem mgh10_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 3;
    problem.num_residuals  = 16;
    problem.residuals      = mgh10_residuals;
    return problem;
}

inline vector_type mgh10_start1()
{
    vector_type b(3);
    b << 2.0, 400000.0, 25000.0;
    return b;
}

inline vector_type mgh10_start2()
{
    vector_type b(3);
    b << 0.02, 4000.0, 250.0;
    return b;
}

inline constexpr double mgh10_certified_rss = 8.7945855171e+01;

inline vector_type mgh10_certified_parameters()
{
    vector_type b(3);
    b << 5.6096364710e-03, 6.1813463463e+03, 3.4522363462e+02;
    return b;
}
}  // namespace solverslib::test
