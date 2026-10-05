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

inline vector_type rosenbrock_minimum()
{
    vector_type x(2);
    x << 1.0, 1.0;
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
// Curve derivatives for RNC-LM solver. Computes derivatives of r(x + t*v + t^2*w + ...)
// along a Taylor expansion centered at x with coefficients [v, w, ...].
inline void rosenbrock_curve_derivatives(const vector_type&                      x,
    const std::vector<vector_type>&                                              coefficients,
    int                                                                           order,
    api::rnc_curve_derivatives&                                                  out)
{
    // Only support orders 1 and 2 for simplicity
    if (order < 1 || order > 2)
    {
        out.residual.clear();
        out.jacobian.clear();
        return;
    }

    out.residual.resize(order + 1);
    out.jacobian.resize(std::max(0, order - 2) + 1);

    // r_0 = r(x)
    out.residual[0].resize(2);
    rosenbrock_residuals(x, out.residual[0]);

    if (order >= 1)
    {
        // J_0 = J(x)
        out.jacobian[0].resize(2, 2);
        rosenbrock_jacobian(x, out.jacobian[0]);

        // r_1 = J(x) * v  (first derivative along direction v = coefficients[0])
        out.residual[1].resize(2);
        out.residual[1] = out.jacobian[0] * coefficients[0];
    }

    if (order >= 2 && coefficients.size() >= 2)
    {
        // For the second derivative, we need ∂²r/∂x²
        // For Rosenbrock: r = [10(x2 - x1²), 1 - x1]
        // Only Hessian of r[0] is non-zero: ∂²r[0]/∂x1² = -20

        // r_2 = 0.5 * H(x)[v, v] + J(x) * w
        // where v = coefficients[0], w = coefficients[1]
        out.residual[2].resize(2);
        out.residual[2].setZero();

        // Hessian-vector product: H[v,v] for r[0]
        const double v1                = coefficients[0](0);
        const double hessian_r0_term   = -20.0;  // ∂²r[0]/∂x1²
        out.residual[2](0) = 0.5 * hessian_r0_term * v1 * v1;  // Only r[0] has non-zero Hessian

        // Add Jacobian term: J(x) * w
        out.residual[2] += out.jacobian[0] * coefficients[1];
    }
}

inline api::least_squares_problem rosenbrock_least_squares_with_rnc(bool with_jacobian)
{
    auto problem = rosenbrock_least_squares(with_jacobian);
    problem.curve_derivatives = std::optional<api::rnc_derivative_function>(rosenbrock_curve_derivatives);
    return problem;
}
}  // namespace solverslib::test
