#include "solvers/api/derivative_provider.h"

#include <sstream>
#include <stdexcept>

#include "detail/eigen_support.h"
#include "detail/support.h"

namespace solverslib::api
{

// ---------------------------------------------------------------------------
// JacobianProvider default residuals_only
// ---------------------------------------------------------------------------

void JacobianProvider::residuals_only(const vector_type& x, vector_type& residuals) const
{
    matrix_type J = make_matrix(num_residuals(), num_parameters());
    compute(x, residuals, J);
}

// ---------------------------------------------------------------------------
// AnalyticJacobianProvider
// ---------------------------------------------------------------------------

AnalyticJacobianProvider::AnalyticJacobianProvider(
    residual_fn rf, jacobian_fn jf, std::size_t n, std::size_t m)
    : residuals_(std::move(rf)), jacobian_(std::move(jf)), n_(n), m_(m)
{
    SOLVERS_CHECK(residuals_, "AnalyticJacobianProvider: residual function must not be null");
    SOLVERS_CHECK(jacobian_, "AnalyticJacobianProvider: Jacobian function must not be null");
}

void AnalyticJacobianProvider::compute(
    const vector_type& x, vector_type& residuals, matrix_type& jacobian) const
{
    residuals_(x, residuals);
    jacobian_(x, jacobian);
}

void AnalyticJacobianProvider::residuals_only(
    const vector_type& x, vector_type& residuals) const
{
    residuals_(x, residuals);
}

// ---------------------------------------------------------------------------
// FiniteDifferenceJacobianProvider
// ---------------------------------------------------------------------------

FiniteDifferenceJacobianProvider::FiniteDifferenceJacobianProvider(
    residual_fn rf, std::size_t n, std::size_t m, double step)
    : residuals_(std::move(rf)), n_(n), m_(m), step_(step)
{
    SOLVERS_CHECK(residuals_, "FiniteDifferenceJacobianProvider: residual function must not be null");
    SOLVERS_CHECK(step > 0.0, "FiniteDifferenceJacobianProvider: step must be positive");
}

void FiniteDifferenceJacobianProvider::compute(
    const vector_type& x, vector_type& residuals, matrix_type& jacobian) const
{
    residuals_(x, residuals);

    vector_type x_plus  = x;
    vector_type x_minus = x;
    vector_type r_plus  = make_vector(m_);
    vector_type r_minus = make_vector(m_);

    for (std::size_t j = 0; j < n_; ++j)
    {
        const double h  = step_ * (std::abs(x[j]) + 1.0);
        x_plus[j]       = x[j] + h;
        x_minus[j]      = x[j] - h;
        residuals_(x_plus, r_plus);
        residuals_(x_minus, r_minus);
        for (std::size_t i = 0; i < m_; ++i)
        {
            jacobian(i, j) = (r_plus[i] - r_minus[i]) / (2.0 * h);
        }
        x_plus[j]  = x[j];
        x_minus[j] = x[j];
    }
}

void FiniteDifferenceJacobianProvider::residuals_only(
    const vector_type& x, vector_type& residuals) const
{
    residuals_(x, residuals);
}

// ---------------------------------------------------------------------------
// AnalyticGradientProvider
// ---------------------------------------------------------------------------

AnalyticGradientProvider::AnalyticGradientProvider(gradient_fn gf, std::size_t n)
    : gradient_(std::move(gf)), n_(n)
{
    SOLVERS_CHECK(gradient_, "AnalyticGradientProvider: gradient function must not be null");
}

void AnalyticGradientProvider::compute(const vector_type& x, vector_type& gradient) const
{
    gradient_(x, gradient);
}

// ---------------------------------------------------------------------------
// FiniteDifferenceGradientProvider
// ---------------------------------------------------------------------------

FiniteDifferenceGradientProvider::FiniteDifferenceGradientProvider(
    objective_fn of, std::size_t n, double step)
    : objective_(std::move(of)), n_(n), step_(step)
{
    SOLVERS_CHECK(objective_, "FiniteDifferenceGradientProvider: objective function must not be null");
    SOLVERS_CHECK(step > 0.0, "FiniteDifferenceGradientProvider: step must be positive");
}

void FiniteDifferenceGradientProvider::compute(
    const vector_type& x, vector_type& gradient) const
{
    vector_type x_plus  = x;
    vector_type x_minus = x;

    for (std::size_t i = 0; i < n_; ++i)
    {
        const double h = step_ * (std::abs(x[i]) + 1.0);
        x_plus[i]      = x[i] + h;
        x_minus[i]     = x[i] - h;
        const double f_plus  = objective_(x_plus);
        const double f_minus = objective_(x_minus);
        gradient[i]          = (f_plus - f_minus) / (2.0 * h);
        x_plus[i]            = x[i];
        x_minus[i]           = x[i];
    }
}

// ---------------------------------------------------------------------------
// Derivative validation
// ---------------------------------------------------------------------------

check_jacobian_result check_jacobian(
    const JacobianProvider& provider_a,
    const JacobianProvider& provider_b,
    const vector_type&      x,
    double                  tol)
{
    SOLVERS_CHECK(
        provider_a.num_parameters() == provider_b.num_parameters() &&
            provider_a.num_residuals() == provider_b.num_residuals(),
        "check_jacobian: providers have incompatible dimensions");

    const std::size_t n = provider_a.num_parameters();
    const std::size_t m = provider_a.num_residuals();

    vector_type r_a = make_vector(m);
    vector_type r_b = make_vector(m);
    matrix_type J_a = make_matrix(m, n);
    matrix_type J_b = make_matrix(m, n);

    provider_a.compute(x, r_a, J_a);
    provider_b.compute(x, r_b, J_b);

    check_jacobian_result result;
    result.passed = true;

    // Check for invalid dimensions
    if (J_a.rows() != static_cast<int>(m) || J_a.cols() != static_cast<int>(n) ||
        J_b.rows() != static_cast<int>(m) || J_b.cols() != static_cast<int>(n))
    {
        result.passed = false;
        result.summary = "FAIL invalid dimensions";
        return result;
    }

    // Independently compute max absolute and max relative errors
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double a = J_a(i, j);
            const double b = J_b(i, j);

            // Check for NaN or Inf in either Jacobian
            if (!std::isfinite(a) || !std::isfinite(b))
            {
                result.passed = false;
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

            // Track worst absolute error independently
            if (abs_err > max_abs_error)
            {
                max_abs_error = abs_err;
                result.worst_row = static_cast<int>(i);
                result.worst_col = static_cast<int>(j);
            }

            // Track worst relative error independently
            if (rel_err > max_rel_error)
            {
                max_rel_error = rel_err;
            }
        }
    }

    result.max_abs_error = max_abs_error;
    result.max_rel_error = max_rel_error;
    result.passed = result.max_abs_error <= tol;

    std::ostringstream oss;
    oss << (result.passed ? "PASS" : "FAIL")
        << " max_abs=" << result.max_abs_error
        << " max_rel=" << result.max_rel_error;
    if (!result.passed)
    {
        oss << " at J(" << result.worst_row << "," << result.worst_col << ")";
    }
    result.summary = oss.str();

    return result;
}

check_gradient_result check_gradient(
    const GradientProvider& provider_a,
    const GradientProvider& provider_b,
    const vector_type&      x,
    double                  tol)
{
    SOLVERS_CHECK(
        provider_a.num_parameters() == provider_b.num_parameters(),
        "check_gradient: providers have incompatible dimensions");

    const std::size_t n = provider_a.num_parameters();

    vector_type g_a = make_vector(n);
    vector_type g_b = make_vector(n);

    provider_a.compute(x, g_a);
    provider_b.compute(x, g_b);

    check_gradient_result result;
    result.passed = true;

    // Check for invalid dimensions
    if (static_cast<std::size_t>(g_a.size()) != n || static_cast<std::size_t>(g_b.size()) != n)
    {
        result.passed = false;
        result.summary = "FAIL invalid dimensions";
        return result;
    }

    // Independently compute max absolute and max relative errors
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        const double a = g_a[i];
        const double b = g_b[i];

        // Check for NaN or Inf in either gradient
        if (!std::isfinite(a) || !std::isfinite(b))
        {
            result.passed = false;
            result.worst_index = static_cast<int>(i);
            std::ostringstream oss;
            oss << "FAIL nonfinite at g[" << i << "]: a=" << a << ", b=" << b;
            result.summary = oss.str();
            return result;
        }

        const double abs_err = std::abs(a - b);
        const double denom   = std::max(std::abs(a), std::abs(b));
        const double rel_err = (denom > 1e-14) ? abs_err / denom : abs_err;

        // Track worst absolute error independently
        if (abs_err > max_abs_error)
        {
            max_abs_error = abs_err;
            result.worst_index = static_cast<int>(i);
        }

        // Track worst relative error independently
        if (rel_err > max_rel_error)
        {
            max_rel_error = rel_err;
        }
    }

    result.max_abs_error = max_abs_error;
    result.max_rel_error = max_rel_error;
    result.passed = result.max_abs_error <= tol;

    std::ostringstream oss;
    oss << (result.passed ? "PASS" : "FAIL")
        << " max_abs=" << result.max_abs_error
        << " max_rel=" << result.max_rel_error;
    if (!result.passed)
    {
        oss << " at g[" << result.worst_index << "]";
    }
    result.summary = oss.str();

    return result;
}

}  // namespace solverslib::api
