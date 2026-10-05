#ifndef SOLVERS_PROBLEM_H_
#define SOLVERS_PROBLEM_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solvers/api/derivative_provider.h"
#include "solvers/api/status.h"
#include "solvers/rnc_lm_derivatives.h"

namespace solverslib::api::detail
{
class provider_factory;
}

namespace solverslib::api
{
// Callback vocabulary. Residual/Jacobian follow the existing native solver
// signatures so the native backend can adopt them without copies. The scalar
// objective/gradient/Hessian family describes general optimization; it stays a
// distinct contract from residual least squares (review: "Keep general scalar
// objectives separate").
using residual_function = std::function<void(const vector_type&, vector_type&)>;
using jacobian_function = std::function<void(const vector_type&, matrix_type&)>;

using objective_function = std::function<double(const vector_type&)>;
using gradient_function  = std::function<void(const vector_type&, vector_type&)>;
using hessian_function   = std::function<void(const vector_type&, matrix_type&)>;
// Matrix-free Hessian-vector product H(x) * v -> out. Its presence steers the
// dispatcher toward a matrix-free (TAO/Newton-Krylov) backend even below the
// size thresholds (review section 7).
using hessian_vector_function =
    std::function<void(const vector_type& x, const vector_type& v, vector_type& out)>;

// Optional box constraints. Either side may be provided independently; an empty
// vector means "no bound on that side" (review F01: one-sided bounds must not be
// silently discarded).
struct bounds
{
    std::vector<double> lower;
    std::vector<double> upper;

    bool empty() const noexcept { return lower.empty() && upper.empty(); }
    bool has_lower() const noexcept { return !lower.empty(); }
    bool has_upper() const noexcept { return !upper.empty(); }
};

// General nonlinear constraints. Only their presence is modeled here; the first
// implementation slice rejects them on backends that cannot enforce them.
struct constraints
{
    std::size_t num_equality   = 0;
    std::size_t num_inequality = 0;

    bool empty() const noexcept { return num_equality == 0 && num_inequality == 0; }
};

// F(x) = 0.5 * ||r(x)||^2, J(i,j) = d r_i / d x_j, g = J^T r.
//
// A problem carries exactly one derivative slot: a JacobianProvider, which
// advertises its capabilities (source(), ceres_factory(), curve_derivatives()).
// A plain Jacobian callback, Ceres-native AD and RNC curve derivatives are all
// providers. In PyTorch a tensor likewise carries one grad_fn and capabilities
// are discovered from it rather than from parallel fields.
struct least_squares_problem
{
    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;

    residual_function residuals;
    api::bounds       bounds;

    // -- Derivative slot ------------------------------------------------------

    // Attach a derivative provider. Replaces any previously set provider.
    void set_jacobian_provider(const std::shared_ptr<JacobianProvider>& provider)
    {
        provider_ = provider;
    }

    // Fluent alias for set_jacobian_provider.
    void derivatives(const std::shared_ptr<JacobianProvider>& provider)
    {
        set_jacobian_provider(provider);
    }

    // Overload for the no-arg auto_diff() sentinel.
    // Requires a model factory (set by least_squares(model, n, m)).
    void derivatives(api::auto_diff_tag)
    {
        if (!model_provider_factory_)
        {
            throw std::invalid_argument(
                "derivatives(auto_diff()) requires a templated model. "
                "Create the problem with least_squares(model, n, m), "
                "or call set_jacobian_provider(auto_diff(model, n, m)) explicitly.");
        }
        set_jacobian_provider(model_provider_factory_());
    }

    // Sugar for analytic_jacobian(residuals, jacobian, n, m). `residuals` must
    // already be set, because the provider evaluates both together.
    void set_jacobian(jacobian_function jacobian_callback)
    {
        if (!residuals)
        {
            throw std::invalid_argument("set_jacobian() requires the residuals to be set first");
        }
        set_jacobian_provider(analytic_jacobian(
            residuals, std::move(jacobian_callback), num_parameters, num_residuals));
    }

    // Derivatives along a curve for RNC-LM (analytic, or Taylor-mode AD with
    // `source == automatic_differentiation`). Also serves as an ordinary
    // Jacobian source for every other backend.
    void set_curve_derivatives(
        rnc_derivative_function derivatives, derivative_mode source = derivative_mode::supplied)
    {
        set_jacobian_provider(
            curve_derivatives(std::move(derivatives), source, num_parameters, num_residuals));
    }

    // Internal: installed by least_squares(model, n, m) for derivatives(auto_diff()).
    void set_model_provider_factory(std::function<std::shared_ptr<JacobianProvider>()> factory)
    {
        model_provider_factory_ = std::move(factory);
    }

    // The attached provider, or null.
    std::shared_ptr<JacobianProvider> derivative_provider() const { return provider_; }

    // -- Queries ----------------------------------------------------------------

    bool has_jacobian_provider() const { return derivative_provider() != nullptr; }

    // True when the Jacobian comes from user-supplied code (callback or analytic
    // provider) rather than AD or finite differences.
    bool has_callable_jacobian() const
    {
        const auto provider = derivative_provider();
        return provider && provider->source() == derivative_mode::supplied;
    }

private:
    std::shared_ptr<JacobianProvider>                  provider_;
    std::function<std::shared_ptr<JacobianProvider>()> model_provider_factory_;
};

// General objective min f(x). One gradient slot, like least_squares_problem.
struct optimization_problem
{
    std::size_t num_parameters = 0;

    objective_function                     objective;
    std::optional<hessian_function>        hessian;
    std::optional<hessian_vector_function> hessian_vector;

    api::bounds      bounds;
    api::constraints constraints;

    // Attach a gradient provider. Replaces any previously set provider.
    void set_gradient_provider(std::shared_ptr<GradientProvider> provider)
    {
        provider_ = std::move(provider);
    }

    // Fluent alias for set_gradient_provider.
    void derivatives(std::shared_ptr<GradientProvider> provider)
    {
        set_gradient_provider(std::move(provider));
    }

    // Sugar for analytic_gradient(gradient, n).
    void set_gradient(gradient_function gradient_callback)
    {
        set_gradient_provider(analytic_gradient(std::move(gradient_callback), num_parameters));
    }

    // The attached provider, or null.
    std::shared_ptr<GradientProvider> derivative_provider() const { return provider_; }

private:
    std::shared_ptr<GradientProvider> provider_;
};

// Structural summary derived from a problem. Drives backend selection so the
// backend never has to be the primary abstraction (review section 2).
struct problem_traits
{
    bool is_least_squares           = false;
    bool has_jacobian               = false;  // any Jacobian source (callback, provider)
    bool has_callable_jacobian      = false;  // explicit analytic jacobian callback
    bool has_jacobian_provider      = false;  // JacobianProvider set
    bool has_autodiff_provider      = false;  // Ceres-native AD factory present
    bool has_gradient               = false;  // any gradient source (callback, provider)
    bool has_gradient_provider      = false;  // GradientProvider set
    bool has_hessian                = false;
    bool has_hessian_vector_product = false;
    bool has_bounds                 = false;
    bool has_nonlinear_constraints  = false;

    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;
};

// One explicit policy for "large scale" rather than magic thresholds scattered
// through the dispatcher (review section 7).
struct dispatch_policy
{
    std::size_t large_parameter_threshold = 1000;
    std::size_t large_residual_threshold  = 10000;
    bool        prefer_matrix_free        = true;
};
}  // namespace solverslib::api

#endif  // SOLVERS_PROBLEM_H_
