#pragma once

#include <cmath>

#include "api/dispatch.h"

// Eckerle4 (NIST StRD nonlinear regression archive).
// Data and certified values from
// https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Eckerle4.dat

namespace solverslib::test
{
inline const double* eckerle4_y()
{
    static const double y[35] = {
    0.0001575, 0.0001699, 0.000235, 0.0003102, 0.0004917, 0.000871, 0.0017418, 0.00464, 
    0.0065895, 0.0097302, 0.0149002, 0.023731, 0.0401683, 0.0712559, 0.1264458, 0.2073413, 
    0.2902366, 0.3445623, 0.3698049, 0.3668534, 0.3106727, 0.2078154, 0.1164354, 0.0616764, 
    0.03372, 0.0194023, 0.0117831, 0.0074357, 0.0022732, 0.00088, 0.0004579, 0.0002345, 
    0.0001586, 0.0001143, 7.1e-05
    };
    return y;
}

inline const double* eckerle4_x()
{
    static const double x[35] = {
    400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 436.5, 438.0, 439.5, 441.0, 442.5, 
    444.0, 445.5, 447.0, 448.5, 450.0, 451.5, 453.0, 454.5, 456.0, 457.5, 459.0, 460.5, 462.0, 
    463.5, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0
    };
    return x;
}

struct eckerle4_model
{
    template <class T> void operator()(const T* b, T* r) const
    {
        using std::exp;
        const double* y = eckerle4_y();
        const double* x_data = eckerle4_x();
        for (int i = 0; i < 35; ++i)
        {
            const T x = T(x_data[i]);
            r[i] = T(y[i]) - ((b[0]/b[1]) * exp(T(-0.5)*((x-b[2])/b[1])*((x-b[2])/b[1])));
        }
    }
};

inline void eckerle4_residuals(const vector_type& b, vector_type& r)
{
    r.resize(35);
    eckerle4_model{}(b.data(), r.data());
}

inline api::least_squares_problem eckerle4_least_squares()
{
    api::least_squares_problem problem;
    problem.num_parameters = 3;
    problem.num_residuals  = 35;
    problem.residuals      = eckerle4_residuals;
    return problem;
}

inline vector_type eckerle4_start1()
{
    vector_type b(3);
    b << 1.0, 10.0, 500.0;
    return b;
}

inline vector_type eckerle4_start2()
{
    vector_type b(3);
    b << 1.5, 5.0, 450.0;
    return b;
}

inline constexpr double eckerle4_certified_rss = 0.0014635887487;

inline vector_type eckerle4_certified_parameters()
{
    vector_type b(3);
    b << 1.5543827178, 4.0888321754, 451.54121844;
    return b;
}
}  // namespace solverslib::test
