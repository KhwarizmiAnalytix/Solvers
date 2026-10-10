#ifndef SOLVERS_DISPATCH_H_
#define SOLVERS_DISPATCH_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solver_options/solver_options.h"
#include "api/result.h"
#include "api/status.h"

namespace solverslib::api
{
// ---------------------------------------------------------------------------
// Derivative types (function-based, not classes)
// ---------------------------------------------------------------------------

// Gradient: g(j) = ∂f / ∂x_j (n-dim vector)
using gradient_function = std::function<void(const vector_type&, vector_type&)>;

// RNC curve derivatives
struct rnc_curve_derivatives
{
    std::vector<vector_type> residual;  // R[0..order]
    std::vector<matrix_type> jacobian;  // J[0..max(0,order-2)]
};

using rnc_derivative_function = std::function<void(
    const vector_type& base,
    const std::vector<vector_type>& coefficients,
    int order,
    rnc_curve_derivatives& out)>;

// ---------------------------------------------------------------------------
// Callback vocabulary
// ---------------------------------------------------------------------------

using residual_function = std::function<void(const vector_type&, vector_type&)>;
using objective_function = std::function<double(const vector_type&)>;
// ---------------------------------------------------------------------------
// Problem constraints and bounds
// ---------------------------------------------------------------------------

struct bounds
{
    std::vector<double> lower;
    std::vector<double> upper;

    bool empty() const noexcept { return lower.empty() && upper.empty(); }
    bool has_lower() const noexcept { return !lower.empty(); }
    bool has_upper() const noexcept { return !upper.empty(); }
};

struct constraints
{
    std::size_t num_equality   = 0;
    std::size_t num_inequality = 0;

    bool empty() const noexcept { return num_equality == 0 && num_inequality == 0; }
};

// ---------------------------------------------------------------------------
// Problem definitions (plain aggregates)
// ---------------------------------------------------------------------------

// F(x) = 0.5 * ||r(x)||^2, J(i,j) = d r_i / d x_j, g = J^T r
struct least_squares_problem
{
    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;

    residual_function residuals;

    // Optional: Jacobian callback. If absent, finite differences will be used.
    std::optional<jacobian_type> jacobian;

    // Optional: Derivatives along a curve for RNC-LM (analytic or Taylor AD).
    std::optional<rnc_derivative_function> curve_derivatives;

    // Qualified so the member name does not hide the type (GCC -Wchanges-meaning).
    solverslib::api::bounds bounds;
};

// General objective min f(x)
struct optimization_problem
{
    std::size_t num_parameters = 0;

    objective_function                     objective;
    std::optional<jacobian_type>           hessian;

    // Optional: gradient callback. If absent, finite differences will be used.
    std::optional<gradient_function> gradient;

    solverslib::api::bounds      bounds;
    solverslib::api::constraints constraints;
};


// Structural summary for backend selection
struct problem_traits
{
    bool is_least_squares           = false;
    bool has_jacobian               = false;
    bool has_gradient               = false;
    bool has_hessian                = false;
    bool has_bounds                 = false;
    bool has_nonlinear_constraints  = false;

    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;
};

// ---------------------------------------------------------------------------
// API entry points
// ---------------------------------------------------------------------------

problem_traits inspect(const least_squares_problem& problem);
problem_traits inspect(const optimization_problem& problem);

solver_result solve(const least_squares_problem& problem,
    const vector_type&                           initial_guess,
    const solver_options&                        options);

solver_result solve(const optimization_problem& problem,
    const vector_type&                          initial_guess,
    const solver_options&                       options);

}  // namespace solverslib::api

#endif  // SOLVERS_DISPATCH_H_
