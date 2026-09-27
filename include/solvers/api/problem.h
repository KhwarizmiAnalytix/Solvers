#ifndef SOLVERS_PROBLEM_H_
#define SOLVERS_PROBLEM_H_

#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solvers/api/derivative_provider.h"

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
struct least_squares_problem
{
    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;

    residual_function                residuals;
    std::optional<jacobian_function> jacobian;

    // Derivative provider (preferred path). Set via set_jacobian_provider() or
    // derivatives(). Accepted by all backends; Ceres uses its native AD path
    // when the provider wraps an AutoDiffJacobianProvider.
    std::shared_ptr<JacobianProvider> jacobian_provider;

    // Stored model factory for use with the no-arg auto_diff() sentinel.
    // Populated by the least_squares(model, n, m) convenience factory.
    // Calling derivatives(auto_diff()) invokes this factory.
    std::function<std::shared_ptr<JacobianProvider>()> model_provider_factory;

    // Ceres-capable backend factory (set alongside jacobian_provider when
    // the provider supports native Ceres AD). Used by the Ceres backend to
    // bypass the generic JacobianProvider interface for better efficiency.
    std::shared_ptr<const detail::provider_factory> provider_factory;

    api::bounds bounds;

    // -- Setters -------------------------------------------------------------

    // Attach a derivative provider. Replaces any previously set provider.
    // This is the recommended way to configure derivatives.
    void set_jacobian_provider(std::shared_ptr<JacobianProvider> provider)
    {
        jacobian_provider = provider;
        if (provider)
        {
            provider_factory = provider->ceres_factory();
        }
        else
        {
            provider_factory = nullptr;
        }
    }

    // Fluent alias for set_jacobian_provider.
    void derivatives(std::shared_ptr<JacobianProvider> provider)
    {
        set_jacobian_provider(std::move(provider));
    }

    // Overload for the no-arg auto_diff() sentinel.
    // Requires model_provider_factory to be set (e.g. via least_squares()).
    void derivatives(api::auto_diff_tag)
    {
        if (!model_provider_factory)
        {
            throw std::invalid_argument(
                "derivatives(auto_diff()) requires a templated model. "
                "Create the problem with least_squares(model, n, m), "
                "or call set_jacobian_provider(auto_diff(model, n, m)) explicitly.");
        }
        set_jacobian_provider(model_provider_factory());
    }

    // -- Queries -------------------------------------------------------------

    bool has_callable_jacobian() const noexcept
    {
        return jacobian.has_value() && static_cast<bool>(jacobian.value());
    }

    bool has_jacobian_provider() const noexcept
    {
        return jacobian_provider != nullptr;
    }

    // -- Deprecated legacy AD interface -------------------------------------

    // DEPRECATED: Use set_jacobian_provider() instead.
    // Type-erased storage for the legacy Ceres AD path. Retained for
    // backward compatibility only; cannot instantiate template member
    // functions from a non-templated context.
    std::any                             templated_residuals;
    std::optional<const std::type_info*> templated_residuals_type;

    template <typename Functor>
    [[deprecated("Use set_jacobian_provider(auto_diff(functor, n, m)) instead")]]
    void set_templated_residuals(const Functor& func)
    {
        templated_residuals      = func;
        templated_residuals_type = &typeid(Functor);
    }

    template <typename Functor>
    [[deprecated("Use jacobian_provider instead")]]
    const Functor* get_templated_residuals() const
    {
        if (!templated_residuals_type.has_value())
        {
            return nullptr;
        }
        if (templated_residuals_type.value() != &typeid(Functor))
        {
            return nullptr;
        }
        try
        {
            return &std::any_cast<const Functor&>(templated_residuals);
        }
        catch (const std::bad_any_cast&)
        {
            return nullptr;
        }
    }

    [[deprecated("Use has_jacobian_provider() instead")]]
    bool has_templated_residuals() const { return templated_residuals_type.has_value(); }
};

// General objective min f(x).
struct optimization_problem
{
    std::size_t num_parameters = 0;

    objective_function                     objective;
    std::optional<gradient_function>       gradient;
    std::optional<hessian_function>        hessian;
    std::optional<hessian_vector_function> hessian_vector;

    // Gradient provider (preferred path). Solvers use this when set.
    // Overrides the gradient callback if both are present.
    std::shared_ptr<GradientProvider> gradient_provider;

    api::bounds      bounds;
    api::constraints constraints;

    // Attach a gradient provider.
    void set_gradient_provider(std::shared_ptr<GradientProvider> provider)
    {
        gradient_provider = std::move(provider);
    }

    // Fluent alias for set_gradient_provider.
    void derivatives(std::shared_ptr<GradientProvider> provider)
    {
        set_gradient_provider(std::move(provider));
    }
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
