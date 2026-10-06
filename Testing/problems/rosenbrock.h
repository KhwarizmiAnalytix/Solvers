#pragma once

#include "api/dispatch.h"

// Rosenbrock function as a least-squares problem and as a scalar optimization
// problem. Both share the minimum x* = (1, 1) with F(x*) = 0.
//   r(x) = [10 (x2 - x1^2), 1 - x1],  F(x) = 0.5 ||r(x)||^2

namespace solverslib::test
{
inline vector_type rosenbrock_start()
{
    vector_type x(2);
    x << -1.2, 1.0;
    return x;
}

inline void rosenbrock_residuals(const vector_type& x, vector_type& r)
{
    r.resize(2);
    r(0) = 10.0 * (x(1) - x(0) * x(0));
    r(1) = 1.0 - x(0);
}

inline void rosenbrock_jacobian(const vector_type& x, matrix_type& j)
{
    j.resize(2, 2);
    j(0, 0) = -20.0 * x(0);
    j(0, 1) = 10.0;
    j(1, 0) = -1.0;
    j(1, 1) = 0.0;
}

inline api::least_squares_problem rosenbrock_least_squares(bool with_jacobian)
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 2;
    problem.residuals      = rosenbrock_residuals;
    if (with_jacobian)
    {
        problem.jacobian = rosenbrock_jacobian;
    }
    return problem;
}

inline double rosenbrock_objective(const vector_type& x)
{
    vector_type r;
    rosenbrock_residuals(x, r);
    return 0.5 * r.squaredNorm();
}

inline void rosenbrock_gradient(const vector_type& x, vector_type& g)
{
    vector_type r;
    matrix_type j;
    rosenbrock_residuals(x, r);
    rosenbrock_jacobian(x, j);
    g = j.transpose() * r;
}

inline api::optimization_problem rosenbrock_optimization(bool with_gradient)
{
    api::optimization_problem problem;
    problem.num_parameters = 2;
    problem.objective      = rosenbrock_objective;
    if (with_gradient)
    {
        problem.gradient = rosenbrock_gradient;
    }
    return problem;
}
}  // namespace solverslib::test
