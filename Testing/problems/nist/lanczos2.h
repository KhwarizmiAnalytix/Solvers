#pragma once

#include <cmath>

#include "api/dispatch.h"

// Lanczos2 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Lanczos2.dat

namespace solverslib::test
{
inline const double* lanczos2_y()
{
    static const double y[24] = {
    2.5134, 2.04433, 1.6684, 1.36642, 1.12323, 0.92689, 0.767934, 0.638878, 0.533784, 0.447936, 
    0.377585, 0.319739, 0.272013, 0.232497, 0.199659, 0.17227, 0.149341, 0.13007, 0.113812, 
    0.100042, 0.0883321, 0.0783354, 0.0697669, 0.0623931
    };
    return y;
}

inline const double* lanczos2_x()
{
    static const double x[24] = {
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
    0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15
    };
    return x;
}

struct lanczos2_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = lanczos2_y();
        const double* x_data = lanczos2_x();
        for (int i = 0; i < 24; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0]*exp(-b[1]*x) + b[2]*exp(-b[3]*x) + b[4]*exp(-b[5]*x));
        }
    }
};

inline void lanczos2_residuals(const vector_type& b, vector_type& r)
{
    r.resize(24);
    lanczos2_model{}(b.data(), r.data());
}

inline api::least_squares_problem lanczos2_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 6;
    problem.num_residuals  = 24;
    problem.residuals      = lanczos2_residuals;
    return problem;
}

inline vector_type lanczos2_start1()
{
    vector_type b(6);
    b << 1.2, 0.3, 5.6, 5.5, 6.5, 7.6;
    return b;
}

inline vector_type lanczos2_start2()
{
    vector_type b(6);
    b << 0.5, 0.7, 3.6, 4.2, 4.0, 6.3;
    return b;
}

inline constexpr double lanczos2_certified_rss = 2.2299428125e-11;

inline vector_type lanczos2_certified_parameters()
{
    vector_type b(6);
    b << 0.096251029939, 1.0057332849, 0.86424689056, 3.0078283915, 1.5529016879, 5.00287981;
    return b;
}
}  // namespace solverslib::test
