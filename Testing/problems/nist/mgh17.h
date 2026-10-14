#pragma once

#include <cmath>

#include "api/dispatch.h"

// MGH17 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/MGH17.dat

namespace solverslib::test
{
inline const double* mgh17_y()
{
    static const double y[33] = {
    0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.85, 0.818, 0.784, 0.751, 0.718, 0.685, 
    0.658, 0.628, 0.603, 0.58, 0.558, 0.538, 0.522, 0.506, 0.49, 0.478, 0.467, 0.457, 0.448, 
    0.438, 0.431, 0.424, 0.42, 0.414, 0.411, 0.406
    };
    return y;
}

inline const double* mgh17_x()
{
    static const double x[33] = {
    0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 
    140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 
    270.0, 280.0, 290.0, 300.0, 310.0, 320.0
    };
    return x;
}

struct mgh17_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = mgh17_y();
        const double* x_data = mgh17_x();
        for (int i = 0; i < 33; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - (b[0] + b[1]*exp(-x*b[3]) + b[2]*exp(-x*b[4]));
        }
    }
};

inline void mgh17_residuals(const vector_type& b, vector_type& r)
{
    r.resize(33);
    mgh17_model{}(b.data(), r.data());
}

inline api::least_squares_problem mgh17_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 5;
    problem.num_residuals  = 33;
    problem.residuals      = mgh17_residuals;
    return problem;
}

inline vector_type mgh17_start1()
{
    vector_type b(5);
    b << 50.0, 150.0, -100.0, 1.0, 2.0;
    return b;
}

inline vector_type mgh17_start2()
{
    vector_type b(5);
    b << 0.5, 1.5, -1.0, 0.01, 0.02;
    return b;
}

inline constexpr double mgh17_certified_rss = 5.4648946975e-05;

inline vector_type mgh17_certified_parameters()
{
    vector_type b(5);
    b << 0.37541005211, 1.9358469127, -1.4646871366, 0.01286753464, 0.022122699662;
    return b;
}
}  // namespace solverslib::test
