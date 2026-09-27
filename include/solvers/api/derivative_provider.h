#ifndef SOLVERS_DERIVATIVE_PROVIDER_H_
#define SOLVERS_DERIVATIVE_PROVIDER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solvers/api/status.h"

// Derivative-provider abstraction layer.
//
// Autodiff, analytic derivatives, and finite differences all implement the
// same JacobianProvider / GradientProvider interface. Solvers consume these
// interfaces without knowing which implementation is in use.
//
// Dependency direction:
//   Solver → Problem → JacobianProvider / GradientProvider → implementation
//
// Never:
//   Solver → AD library

namespace solverslib::api::detail
{
class provider_factory;
}

namespace solverslib::api
{

// ---------------------------------------------------------------------------
// JacobianProvider — r: R^n → R^m
// ---------------------------------------------------------------------------
class JacobianProvider
{
public:
    virtual ~JacobianProvider() = default;

    // Compute residuals and Jacobian simultaneously.
    // residuals: output buffer of size m
    // jacobian:  output m×n matrix (row i, column j = ∂r_i/∂x_j)
    virtual void compute(
        const vector_type& x,
        vector_type&       residuals,
        matrix_type&       jacobian) const = 0;

    // Compute only residuals (no Jacobian). Default delegates to compute().
    virtual void residuals_only(const vector_type& x, vector_type& residuals) const;

    virtual std::size_t     num_parameters() const = 0;
    virtual std::size_t     num_residuals() const  = 0;
    virtual derivative_mode source() const         = 0;

    // Optional: expose an internal Ceres-capable provider factory so the Ceres
    // backend can use its native autodiff path instead of going through this
    // interface. Returns nullptr for all non-Ceres implementations.
    // This does not leak Ceres types into the public interface; provider_factory
    // is a backend-neutral abstract factory.
    virtual std::shared_ptr<const solverslib::api::detail::provider_factory> ceres_factory() const
    {
        return nullptr;
    }
};

// ---------------------------------------------------------------------------
// GradientProvider — f: R^n → R
// ---------------------------------------------------------------------------
class GradientProvider
{
public:
    virtual ~GradientProvider() = default;

    // Compute gradient at x. gradient: output buffer of size n.
    virtual void compute(const vector_type& x, vector_type& gradient) const = 0;

    virtual std::size_t     num_parameters() const = 0;
    virtual derivative_mode source() const         = 0;
};

// ---------------------------------------------------------------------------
// Analytic Jacobian provider — wraps user-supplied residual + Jacobian fns
// ---------------------------------------------------------------------------
class AnalyticJacobianProvider final : public JacobianProvider
{
public:
    using residual_fn = std::function<void(const vector_type&, vector_type&)>;
    using jacobian_fn = std::function<void(const vector_type&, matrix_type&)>;

    AnalyticJacobianProvider(
        residual_fn rf, jacobian_fn jf, std::size_t n, std::size_t m);

    void compute(
        const vector_type& x,
        vector_type&       residuals,
        matrix_type&       jacobian) const override;

    void residuals_only(const vector_type& x, vector_type& residuals) const override;

    std::size_t     num_parameters() const override { return n_; }
    std::size_t     num_residuals() const override { return m_; }
    derivative_mode source() const override { return derivative_mode::supplied; }

private:
    residual_fn residuals_;
    jacobian_fn jacobian_;
    std::size_t n_, m_;
};

// ---------------------------------------------------------------------------
// Finite-difference Jacobian provider — central differences of residuals
// ---------------------------------------------------------------------------
class FiniteDifferenceJacobianProvider final : public JacobianProvider
{
public:
    using residual_fn = std::function<void(const vector_type&, vector_type&)>;

    explicit FiniteDifferenceJacobianProvider(
        residual_fn rf,
        std::size_t n,
        std::size_t m,
        double      step = 1e-7);

    void compute(
        const vector_type& x,
        vector_type&       residuals,
        matrix_type&       jacobian) const override;

    void residuals_only(const vector_type& x, vector_type& residuals) const override;

    std::size_t     num_parameters() const override { return n_; }
    std::size_t     num_residuals() const override { return m_; }
    derivative_mode source() const override { return derivative_mode::finite_difference; }

private:
    residual_fn residuals_;
    std::size_t n_, m_;
    double      step_;
};

// ---------------------------------------------------------------------------
// Analytic gradient provider — wraps user-supplied gradient function
// ---------------------------------------------------------------------------
class AnalyticGradientProvider final : public GradientProvider
{
public:
    using gradient_fn = std::function<void(const vector_type&, vector_type&)>;

    AnalyticGradientProvider(gradient_fn gf, std::size_t n);

    void        compute(const vector_type& x, vector_type& gradient) const override;
    std::size_t num_parameters() const override { return n_; }
    derivative_mode source() const override { return derivative_mode::supplied; }

private:
    gradient_fn gradient_;
    std::size_t n_;
};

// ---------------------------------------------------------------------------
// Finite-difference gradient provider — central differences of objective
// ---------------------------------------------------------------------------
class FiniteDifferenceGradientProvider final : public GradientProvider
{
public:
    using objective_fn = std::function<double(const vector_type&)>;

    explicit FiniteDifferenceGradientProvider(
        objective_fn of, std::size_t n, double step = 1e-7);

    void        compute(const vector_type& x, vector_type& gradient) const override;
    std::size_t num_parameters() const override { return n_; }
    derivative_mode source() const override { return derivative_mode::finite_difference; }

private:
    objective_fn objective_;
    std::size_t  n_;
    double       step_;
};

// ---------------------------------------------------------------------------
// Sentinel type for auto_diff() with no arguments.
// Used with least_squares(model, n, m) or minimize(model, n) which store a
// model-provider factory on the problem. problem.derivatives(auto_diff()) then
// instantiates the actual provider from that stored factory.
// ---------------------------------------------------------------------------
struct auto_diff_tag
{
};

inline auto_diff_tag auto_diff()
{
    return {};
}

// ---------------------------------------------------------------------------
// Factory helpers — convenience wrappers to create providers
// ---------------------------------------------------------------------------

inline std::shared_ptr<AnalyticJacobianProvider> analytic_jacobian(
    std::function<void(const vector_type&, vector_type&)> residual_fn,
    std::function<void(const vector_type&, matrix_type&)> jacobian_fn,
    std::size_t                                           n,
    std::size_t                                           m)
{
    return std::make_shared<AnalyticJacobianProvider>(
        std::move(residual_fn), std::move(jacobian_fn), n, m);
}

inline std::shared_ptr<FiniteDifferenceJacobianProvider> finite_difference(
    std::function<void(const vector_type&, vector_type&)> residual_fn,
    std::size_t                                           n,
    std::size_t                                           m,
    double                                                step = 1e-7)
{
    return std::make_shared<FiniteDifferenceJacobianProvider>(
        std::move(residual_fn), n, m, step);
}

inline std::shared_ptr<AnalyticGradientProvider> analytic_gradient(
    std::function<void(const vector_type&, vector_type&)> gradient_fn,
    std::size_t                                           n)
{
    return std::make_shared<AnalyticGradientProvider>(std::move(gradient_fn), n);
}

inline std::shared_ptr<FiniteDifferenceGradientProvider> finite_difference_gradient(
    std::function<double(const vector_type&)> objective_fn,
    std::size_t                               n,
    double                                    step = 1e-7)
{
    return std::make_shared<FiniteDifferenceGradientProvider>(
        std::move(objective_fn), n, step);
}

// ---------------------------------------------------------------------------
// Derivative validation utilities
// Compares two derivative sources (analytic vs AD, analytic vs FD, etc.)
// and reports the worst discrepancy. Useful for debugging and calibration.
// ---------------------------------------------------------------------------

struct check_jacobian_result
{
    bool   passed             = false;
    double max_abs_error      = 0.0;
    double max_rel_error      = 0.0;
    int    worst_row          = -1;
    int    worst_col          = -1;
    std::string summary;
};

struct check_gradient_result
{
    bool   passed        = false;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int    worst_index   = -1;
    std::string summary;
};

// Compare two Jacobian implementations at a test point.
// provider_a and provider_b must agree on dimensions.
// tol: maximum acceptable absolute error
check_jacobian_result check_jacobian(
    const JacobianProvider& provider_a,
    const JacobianProvider& provider_b,
    const vector_type&      x,
    double                  tol = 1e-5);

// Compare two gradient implementations at a test point.
check_gradient_result check_gradient(
    const GradientProvider& provider_a,
    const GradientProvider& provider_b,
    const vector_type&      x,
    double                  tol = 1e-5);

}  // namespace solverslib::api

#endif  // SOLVERS_DERIVATIVE_PROVIDER_H_
