#pragma once

#include <cmath>

#include "api/dispatch.h"

// Thurber (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Thurber.dat

namespace solverslib::test
{
inline const double* thurber_y()
{
    static const double y[37] = {
    80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101, 92.405, 95.854, 100.696, 
    101.06, 401.672, 390.724, 567.534, 635.316, 733.054, 759.087, 894.206, 990.785, 1090.109, 
    1080.914, 1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509, 
    1425.208, 1421.384, 1442.962, 1464.35, 1468.705, 1447.894, 1457.628
    };
    return y;
}

inline const double* thurber_x()
{
    static const double x[37] = {
    -3.067, -2.981, -2.921, -2.912, -2.84, -2.797, -2.702, -2.699, -2.633, -2.481, -2.363, 
    -2.322, -1.501, -1.46, -1.274, -1.212, -1.1, -1.046, -0.915, -0.714, -0.566, -0.545, -0.4, 
    -0.309, -0.109, -0.103, 0.01, 0.119, 0.377, 0.79, 0.963, 1.006, 1.115, 1.572, 1.841, 2.047, 
    2.2
    };
    return x;
}

struct thurber_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        const double* y = thurber_y();
        const double* x_data = thurber_x();
        for (int i = 0; i < 37; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - ((b[0] + b[1]*x + b[2]*x*x + b[3]*x*x*x) / (T(1) + b[4]*x + b[5]*x*x + b[6]*x*x*x));
        }
    }
};

inline void thurber_residuals(const vector_type& b, vector_type& r)
{
    r.resize(37);
    thurber_model{}(b.data(), r.data());
}

inline api::least_squares_problem thurber_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 7;
    problem.num_residuals  = 37;
    problem.residuals      = thurber_residuals;
    return problem;
}

inline vector_type thurber_start1()
{
    vector_type b(7);
    b << 1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03;
    return b;
}

inline vector_type thurber_start2()
{
    vector_type b(7);
    b << 1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05;
    return b;
}

inline constexpr double thurber_certified_rss = 5642.7082397;

inline vector_type thurber_certified_parameters()
{
    vector_type b(7);
    b << 1288.13968, 1491.0792535, 583.23836877, 75.416644291, 0.96629502864, 0.39797285797, 0.049727297349;
    return b;
}
}  // namespace solverslib::test
