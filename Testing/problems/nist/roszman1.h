#pragma once

#include <cmath>

#include "api/dispatch.h"

// Roszman1 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Roszman1.dat

namespace solverslib::test
{
inline const double* roszman1_y()
{
    static const double y[25] = {
    0.252429, 0.252141, 0.251809, 0.297989, 0.296257, 0.295319, 0.339603, 0.337731, 0.33382, 
    0.38951, 0.386998, 0.438864, 0.434887, 0.427893, 0.471568, 0.461699, 0.461144, 0.513532, 
    0.506641, 0.505062, 0.535648, 0.533726, 0.568064, 0.612886, 0.624169
    };
    return y;
}

inline const double* roszman1_x()
{
    static const double x[25] = {
    -4868.68, -4868.09, -4867.41, -3375.19, -3373.14, -3372.03, -2473.74, -2472.35, -2469.45, 
    -1894.65, -1893.4, -1497.24, -1495.85, -1493.41, -1208.68, -1206.18, -1206.04, -997.92, 
    -996.61, -996.31, -834.94, -834.66, -710.03, -530.16, -464.17
    };
    return x;
}

struct roszman1_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::atan;
        const double* y = roszman1_y();
        const double* x_data = roszman1_x();
        for (int i = 0; i < 25; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0] - b[1]*x - atan(b[2]/(x-b[3])) / T(3.141592653589793238462643383279));
        }
    }
};

inline void roszman1_residuals(const vector_type& b, vector_type& r)
{
    r.resize(25);
    roszman1_model{}(b.data(), r.data());
}

inline api::least_squares_problem roszman1_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 4;
    problem.num_residuals  = 25;
    problem.residuals      = roszman1_residuals;
    return problem;
}

inline vector_type roszman1_start1()
{
    vector_type b(4);
    b << 0.1, -1e-05, 1000.0, -100.0;
    return b;
}

inline vector_type roszman1_start2()
{
    vector_type b(4);
    b << 0.2, -5e-06, 1200.0, -150.0;
    return b;
}

inline constexpr double roszman1_certified_rss = 0.00049484847331;

inline vector_type roszman1_certified_parameters()
{
    vector_type b(4);
    b << 0.20196866396, -6.1953516256e-06, 1204.4556708, -181.34269537;
    return b;
}
}  // namespace solverslib::test
