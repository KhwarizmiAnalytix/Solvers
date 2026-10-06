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
using hessian_vector_function =
    std::function<void(const vector_type& x, const vector_type& v, vector_type& out)>;

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

    bounds bounds;
};

// General objective min f(x)
struct optimization_problem
{
    std::size_t num_parameters = 0;

    objective_function                     objective;
    std::optional<jacobian_type>           hessian;
    std::optional<hessian_vector_function> hessian_vector;

    // Optional: gradient callback. If absent, finite differences will be used.
    std::optional<gradient_function> gradient;

    bounds      bounds;
    constraints constraints;
};

// Structural summary for backend selection
struct problem_traits
{
    bool is_least_squares           = false;
    bool has_jacobian               = false;
    bool has_gradient               = false;
    bool has_hessian                = false;
    bool has_hessian_vector_product = false;
    bool has_bounds                 = false;
    bool has_nonlinear_constraints  = false;

    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;
};

// ---------------------------------------------------------------------------
// Derivative validation utilities
// ---------------------------------------------------------------------------

struct check_jacobian_result
{
    bool        passed        = false;
    double      max_abs_error = 0.0;
    double      max_rel_error = 0.0;
    int         worst_row     = -1;
    int         worst_col     = -1;
    std::string summary;
};

struct check_gradient_result
{
    bool        passed        = false;
    double      max_abs_error = 0.0;
    double      max_rel_error = 0.0;
    int         worst_index   = -1;
    std::string summary;
};

// Compare two Jacobian implementations at x
inline check_jacobian_result check_jacobian(
    const jacobian_type& jacobian_a,
    const jacobian_type& jacobian_b,
    const vector_type&       x,
    std::size_t              m,
    std::size_t              n,
    double                   tol = 1e-5)
{
    matrix_type J_a = make_matrix(m, n);
    matrix_type J_b = make_matrix(m, n);

    try
    {
        jacobian_a(x, J_a);
        jacobian_b(x, J_b);
    }
    catch (const std::exception& e)
    {
        return {false, 0.0, 0.0, -1, -1, std::string("exception: ") + e.what()};
    }

    check_jacobian_result result;
    result.passed = true;

    if (J_a.rows() != static_cast<int>(m) || J_a.cols() != static_cast<int>(n) ||
        J_b.rows() != static_cast<int>(m) || J_b.cols() != static_cast<int>(n))
    {
        result.passed  = false;
        result.summary = "FAIL invalid dimensions";
        return result;
    }

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double a = J_a(i, j);
            const double b = J_b(i, j);

            if (!std::isfinite(a) || !std::isfinite(b))
            {
                result.passed    = false;
                result.worst_row = static_cast<int>(i);
                result.worst_col = static_cast<int>(j);
                std::ostringstream oss;
                oss << "FAIL nonfinite at J(" << i << "," << j << "): a=" << a << ", b=" << b;
                result.summary = oss.str();
                return result;
            }

            const double abs_err = std::abs(a - b);
            const double denom   = std::max(std::abs(a), std::abs(b));
            const double rel_err = (denom > 1e-14) ? abs_err / denom : abs_err;

            if (abs_err > max_abs_error)
            {
                max_abs_error    = abs_err;
                result.worst_row = static_cast<int>(i);
                result.worst_col = static_cast<int>(j);
            }

            if (rel_err > max_rel_error)
            {
                max_rel_error = rel_err;
            }
        }
    }

    result.max_abs_error = max_abs_error;
    result.max_rel_error = max_rel_error;
    result.passed        = result.max_abs_error <= tol;

    std::ostringstream oss;
    oss << (result.passed ? "PASS" : "FAIL") << " max_abs=" << result.max_abs_error
        << " max_rel=" << result.max_rel_error;
    if (!result.passed)
    {
        oss << " at J(" << result.worst_row << "," << result.worst_col << ")";
    }
    result.summary = oss.str();

    return result;
}

// Compare two gradient implementations at x
inline check_gradient_result check_gradient(
    const gradient_function& gradient_a,
    const gradient_function& gradient_b,
    const vector_type&       x,
    std::size_t              n,
    double                   tol = 1e-5)
{
    vector_type g_a = make_vector(n);
    vector_type g_b = make_vector(n);

    try
    {
        gradient_a(x, g_a);
        gradient_b(x, g_b);
    }
    catch (const std::exception& e)
    {
        return {false, 0.0, 0.0, -1, std::string("exception: ") + e.what()};
    }

    check_gradient_result result;
    result.passed = true;

    if (static_cast<std::size_t>(g_a.size()) != n || static_cast<std::size_t>(g_b.size()) != n)
    {
        result.passed  = false;
        result.summary = "FAIL invalid dimensions";
        return result;
    }

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        const double a = g_a[i];
        const double b = g_b[i];

        if (!std::isfinite(a) || !std::isfinite(b))
        {
            result.passed      = false;
            result.worst_index = static_cast<int>(i);
            std::ostringstream oss;
            oss << "FAIL nonfinite at g[" << i << "]: a=" << a << ", b=" << b;
            result.summary = oss.str();
            return result;
        }

        const double abs_err = std::abs(a - b);
        const double denom   = std::max(std::abs(a), std::abs(b));
        const double rel_err = (denom > 1e-14) ? abs_err / denom : abs_err;

        if (abs_err > max_abs_error)
        {
            max_abs_error      = abs_err;
            result.worst_index = static_cast<int>(i);
        }

        if (rel_err > max_rel_error)
        {
            max_rel_error = rel_err;
        }
    }

    result.max_abs_error = max_abs_error;
    result.max_rel_error = max_rel_error;
    result.passed        = result.max_abs_error <= tol;

    std::ostringstream oss;
    oss << (result.passed ? "PASS" : "FAIL") << " max_abs=" << result.max_abs_error
        << " max_rel=" << result.max_rel_error;
    if (!result.passed)
    {
        oss << " at g[" << result.worst_index << "]";
    }
    result.summary = oss.str();

    return result;
}

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
