#pragma once

#include <cmath>

#include "api/dispatch.h"

// Lanczos3 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Lanczos3.dat

namespace solverslib::test
{
inline const double* lanczos3_y()
{
    static const double y[24] = {
    2.5134, 2.0443, 1.6684, 1.3664, 1.1232, 0.9269, 0.7679, 0.6389, 0.5338, 0.4479, 0.3776, 
    0.3197, 0.272, 0.2325, 0.1997, 0.1723, 0.1493, 0.1301, 0.1138, 0.1, 0.0883, 0.0783, 0.0698, 
    0.0624
    };
    return y;
}

inline const double* lanczos3_x()
{
    static const double x[24] = {
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
    0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15
    };
    return x;
}

struct lanczos3_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = lanczos3_y();
        const double* x_data = lanczos3_x();
        for (int i = 0; i < 24; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0]*exp(-b[1]*x) + b[2]*exp(-b[3]*x) + b[4]*exp(-b[5]*x));
        }
    }
};

inline void lanczos3_residuals(const vector_type& b, vector_type& r)
{
    r.resize(24);
    lanczos3_model{}(b.data(), r.data());
}

inline api::least_squares_problem lanczos3_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 6;
    problem.num_residuals  = 24;
    problem.residuals      = lanczos3_residuals;
    return problem;
}

inline vector_type lanczos3_start1()
{
    vector_type b(6);
    b << 1.2, 0.3, 5.6, 5.5, 6.5, 7.6;
    return b;
}

inline vector_type lanczos3_start2()
{
    vector_type b(6);
    b << 0.5, 0.7, 3.6, 4.2, 4.0, 6.3;
    return b;
}

inline constexpr double lanczos3_certified_rss = 1.6117193594e-08;

inline vector_type lanczos3_certified_parameters()
{
    vector_type b(6);
    b << 0.086816414977, 0.95498101505, 0.84400777463, 2.9515951832, 1.5825685901, 4.9863565084;
    return b;
}
}  // namespace solverslib::test
