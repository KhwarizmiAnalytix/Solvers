#ifndef SOLVERS_DERIVATIVE_PROVIDER_H_
#define SOLVERS_DERIVATIVE_PROVIDER_H_

#include <cmath>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solvers/api/status.h"

// Simple function-based derivatives: no virtual classes, no shared_ptr wrapping.
//
// jacobian_function: compute Jacobian J (m x n) at x
// gradient_function: compute gradient g (n-dim) at x
// rnc_derivative_function: Taylor coefficients along a curve (for RNC-LM)

namespace solverslib::api
{

// Jacobian: J(i, j) = ∂r_i / ∂x_j (m x n matrix)
using jacobian_function = std::function<void(const vector_type&, matrix_type&)>;

// Gradient: g(j) = ∂f / ∂x_j (n-dim vector)
using gradient_function = std::function<void(const vector_type&, vector_type&)>;

// RNC curve derivatives: Taylor coefficients of r(x(t)) and J(x(t)) along a curve
struct rnc_curve_derivatives
{
    std::vector<vector_type> residual;  // R[0..order]
    std::vector<matrix_type> jacobian;  // J[0..max(0,order-2)]
};

// RNC curve derivative function
using rnc_derivative_function = std::function<void(
    const vector_type& base,
    const std::vector<vector_type>& coefficients,
    int order,
    rnc_curve_derivatives& out)>;

// ---------------------------------------------------------------------------
// Validation utilities for comparing two jacobian/gradient implementations
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

// Compare two Jacobian implementations at x.
inline check_jacobian_result check_jacobian(
    const jacobian_function& jacobian_a,
    const jacobian_function& jacobian_b,
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

// Compare two gradient implementations at x.
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

}  // namespace solverslib::api

#endif  // SOLVERS_DERIVATIVE_PROVIDER_H_
