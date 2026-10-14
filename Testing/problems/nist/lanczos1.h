#pragma once

#include <cmath>

#include "api/dispatch.h"

// Lanczos1 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Lanczos1.dat

namespace solverslib::test
{
inline const double* lanczos1_y()
{
    static const double y[24] = {
    2.5134, 2.044333373291, 1.668404436564, 1.366418021208, 1.123232487372, 0.9268897180037, 
    0.7679338563728, 0.6388775523106, 0.5337835317402, 0.4479363617347, 0.377584788435, 
    0.3197393199326, 0.2720130773746, 0.2324965529032, 0.1996589546065, 0.1722704126914, 
    0.1493405660168, 0.1300700206922, 0.1138119324644, 0.1000415587559, 0.0883320908454, 
    0.0783354401935, 0.06976693743449, 0.06239312536719
    };
    return y;
}

inline const double* lanczos1_x()
{
    static const double x[24] = {
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
    0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15
    };
    return x;
}

struct lanczos1_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = lanczos1_y();
        const double* x_data = lanczos1_x();
        for (int i = 0; i < 24; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0]*exp(-b[1]*x) + b[2]*exp(-b[3]*x) + b[4]*exp(-b[5]*x));
        }
    }
};

inline void lanczos1_residuals(const vector_type& b, vector_type& r)
{
    r.resize(24);
    lanczos1_model{}(b.data(), r.data());
}

inline api::least_squares_problem lanczos1_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 6;
    problem.num_residuals  = 24;
    problem.residuals      = lanczos1_residuals;
    return problem;
}

inline vector_type lanczos1_start1()
{
    vector_type b(6);
    b << 1.2, 0.3, 5.6, 5.5, 6.5, 7.6;
    return b;
}

inline vector_type lanczos1_start2()
{
    vector_type b(6);
    b << 0.5, 0.7, 3.6, 4.2, 4.0, 6.3;
    return b;
}

inline constexpr double lanczos1_certified_rss = 1.4307867721e-25;

inline vector_type lanczos1_certified_parameters()
{
    vector_type b(6);
    b << 0.095100000027, 1.0000000001, 0.86070000013, 3.0000000002, 1.5575999998, 5.0000000001;
    return b;
}
}  // namespace solverslib::test
