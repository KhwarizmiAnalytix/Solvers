#pragma once

#include <cmath>

#include "api/dispatch.h"

// BoxBOD (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/BoxBOD.dat

namespace solverslib::test
{
inline const double* boxbod_y()
{
    static const double y[6] = {
    109.0, 149.0, 149.0, 191.0, 213.0, 224.0
    };
    return y;
}

inline const double* boxbod_x()
{
    static const double x[6] = {
    1.0, 2.0, 3.0, 5.0, 7.0, 10.0
    };
    return x;
}

struct boxbod_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = boxbod_y();
        const double* x_data = boxbod_x();
        for (int i = 0; i < 6; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0]*(T(1) - exp(-b[1]*x)));
        }
    }
};

inline void boxbod_residuals(const vector_type& b, vector_type& r)
{
    r.resize(6);
    boxbod_model{}(b.data(), r.data());
}

inline api::least_squares_problem boxbod_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 6;
    problem.residuals      = boxbod_residuals;
    return problem;
}

inline vector_type boxbod_start1()
{
    vector_type b(2);
    b << 1.0, 1.0;
    return b;
}

inline vector_type boxbod_start2()
{
    vector_type b(2);
    b << 100.0, 0.75;
    return b;
}

inline constexpr double boxbod_certified_rss = 1168.0088766;

inline vector_type boxbod_certified_parameters()
{
    vector_type b(2);
    b << 213.80940889, 0.54723748542;
    return b;
}
}  // namespace solverslib::test
