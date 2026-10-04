#include "solvers/api/detail/evaluators.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace solverslib::api::detail
{
namespace
{
constexpr double kInfinity = std::numeric_limits<double>::infinity();

double lower_at(const api::bounds& bounds, std::size_t i)
{
    return bounds.has_lower() ? bounds.lower[i] : -kInfinity;
}

double upper_at(const api::bounds& bounds, std::size_t i)
{
    return bounds.has_upper() ? bounds.upper[i] : kInfinity;
}

// One stencil decision per coordinate, shared by Jacobian and gradient.
struct stencil
{
    enum class kind : std::uint8_t
    {
        central,
        forward,
        backward,
        none
    };
    kind   shape = kind::central;
    double h     = 0.0;
};

stencil choose_stencil(double xi, double relative_step, double lo, double hi)
{
    double h = relative_step * std::max(1.0, std::abs(xi));
    if (xi + h <= hi && xi - h >= lo)
    {
        return {stencil::kind::central, h};
    }
    // The interval is too narrow for a central stencil: step toward the roomier
    // side, no further than the interval allows.
    h = std::min(h, hi - lo);
    if (!(h > 0.0))
    {
        return {stencil::kind::none, 0.0};
    }
    return (hi - xi >= xi - lo) ? stencil{stencil::kind::forward, h}
                                : stencil{stencil::kind::backward, h};
}

template <class Callable>
evaluation_status guarded(std::optional<std::string>& error, Callable&& body)
{
    try
    {
        return body();
    }
    catch (const std::exception& e)
    {
        error = e.what();
    }
    catch (...)
    {
        error = "unknown exception thrown by a user callback";
    }
    return evaluation_status::fatal_error;
}

void ensure_size(vector_type& v, std::size_t n)
{
    resize_if_needed(v, n);
}

void ensure_size(matrix_type& a, std::size_t rows, std::size_t cols)
{
    resize_if_needed(a, rows, cols);
}
}  // namespace

// -- finite differences ----------------------------------------------------------
void finite_difference_jacobian(const residual_function& f,
    const vector_type&                                   x,
    const vector_type&                                   residuals_at_x,
    double                                               relative_step,
    const api::bounds&                                   bounds,
    matrix_type&                                         jacobian,
    std::size_t&                                         evaluations,
    finite_difference_workspace*                         workspace)
{
    const auto n = static_cast<std::size_t>(x.size());
    const auto m = static_cast<std::size_t>(residuals_at_x.size());
    ensure_size(jacobian, m, n);

    finite_difference_workspace  local;
    finite_difference_workspace& scratch = workspace != nullptr ? *workspace : local;
    ensure_size(scratch.plus, m);
    ensure_size(scratch.minus, m);
    ensure_size(scratch.probe, n);
    scratch.probe      = x;
    vector_type& probe = scratch.probe;
    vector_type& plus  = scratch.plus;
    vector_type& minus = scratch.minus;
    for (std::size_t j = 0; j < n; ++j)
    {
        const auto   index = static_cast<index_type>(j);
        const double xj    = x[index];
        const auto   shape =
            choose_stencil(xj, relative_step, lower_at(bounds, j), upper_at(bounds, j));
        switch (shape.shape)
        {
        case stencil::kind::central:
            probe[index] = xj + shape.h;
            f(probe, plus);
            probe[index] = xj - shape.h;
            f(probe, minus);
            evaluations += 2;
            jacobian.col(index) = (plus - minus) / (2.0 * shape.h);
            break;
        case stencil::kind::forward:
            probe[index] = xj + shape.h;
            f(probe, plus);
            ++evaluations;
            jacobian.col(index) = (plus - residuals_at_x) / shape.h;
            break;
        case stencil::kind::backward:
            probe[index] = xj - shape.h;
            f(probe, minus);
            ++evaluations;
            jacobian.col(index) = (residuals_at_x - minus) / shape.h;
            break;
        case stencil::kind::none:
            jacobian.col(index).setZero();
            break;
        }
        probe[index] = xj;
    }
}

void finite_difference_gradient(const objective_function& f,
    const vector_type&                                    x,
    const double*                                         value_at_x,
    double                                                relative_step,
    const api::bounds&                                    bounds,
    vector_type&                                          gradient,
    std::size_t&                                          evaluations,
    finite_difference_workspace*                          workspace)
{
    const auto n = static_cast<std::size_t>(x.size());
    ensure_size(gradient, n);

    std::optional<double> centre;
    if (value_at_x != nullptr)
    {
        centre = *value_at_x;
    }
    auto value_here = [&]
    {
        if (!centre)
        {
            centre = f(x);
            ++evaluations;
        }
        return *centre;
    };

    finite_difference_workspace  local;
    finite_difference_workspace& scratch = workspace != nullptr ? *workspace : local;
    ensure_size(scratch.probe, n);
    scratch.probe      = x;
    vector_type& probe = scratch.probe;
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto   index = static_cast<index_type>(i);
        const double xi    = x[index];
        const auto   shape =
            choose_stencil(xi, relative_step, lower_at(bounds, i), upper_at(bounds, i));
        switch (shape.shape)
        {
        case stencil::kind::central:
        {
            probe[index]       = xi + shape.h;
            const double plus  = f(probe);
            probe[index]       = xi - shape.h;
            const double minus = f(probe);
            evaluations += 2;
            gradient[index] = (plus - minus) / (2.0 * shape.h);
            break;
        }
        case stencil::kind::forward:
        {
            probe[index]      = xi + shape.h;
            const double plus = f(probe);
            ++evaluations;
            gradient[index] = (plus - value_here()) / shape.h;
            break;
        }
        case stencil::kind::backward:
        {
            probe[index]       = xi - shape.h;
            const double minus = f(probe);
            ++evaluations;
            gradient[index] = (value_here() - minus) / shape.h;
            break;
        }
        case stencil::kind::none:
            gradient[index] = 0.0;
            break;
        }
        probe[index] = xi;
    }
}

// -- callback_residual_evaluator ---------------------------------------------------
callback_residual_evaluator::callback_residual_evaluator(std::size_t n,
    std::size_t                                                      m,
    residual_function                                                residuals,
    std::optional<jacobian_function>                                 jacobian)
    : residuals_(std::move(residuals)), jacobian_(std::move(jacobian)),
      metadata_{n, m, api::derivative_mode::supplied, false}
{
}

evaluation_status callback_residual_evaluator::do_evaluate(
    const vector_type& x, vector_type& residuals, matrix_type* jacobians)
{
    return guarded(last_error_,
        [&]
        {
            ensure_size(residuals, metadata_.num_residuals);
            residuals_(x, residuals);
            if (!residuals.allFinite())
            {
                return evaluation_status::invalid_trial;
            }
            if (jacobians != nullptr)
            {
                if (!jacobian_ || !*jacobian_)
                {
                    last_error_ = "no Jacobian callback available";
                    return evaluation_status::fatal_error;
                }
                ensure_size(*jacobians, metadata_.num_residuals, metadata_.num_parameters);
                (*jacobian_)(x, *jacobians);
                if (!jacobians->allFinite())
                {
                    return evaluation_status::invalid_trial;
                }
            }
            return evaluation_status::ok;
        });
}

evaluation_status callback_residual_evaluator::do_jacobian(
    const vector_type& x, const vector_type& /*residuals_at_x*/, matrix_type& jacobian)
{
    return guarded(last_error_,
        [&]
        {
            if (!jacobian_ || !*jacobian_)
            {
                last_error_ = "no Jacobian callback available";
                return evaluation_status::fatal_error;
            }
            ensure_size(jacobian, metadata_.num_residuals, metadata_.num_parameters);
            (*jacobian_)(x, jacobian);
            return jacobian.allFinite() ? evaluation_status::ok : evaluation_status::invalid_trial;
        });
}

// -- finite_difference_residual_evaluator -----------------------------------------
finite_difference_residual_evaluator::finite_difference_residual_evaluator(std::size_t n,
    std::size_t                                                                        m,
    residual_function                                                                  residuals,
    double      relative_step,
    api::bounds bounds)
    : residuals_(std::move(residuals)), relative_step_(relative_step), bounds_(std::move(bounds)),
      metadata_{n, m, api::derivative_mode::finite_difference, false}
{
}

evaluation_status finite_difference_residual_evaluator::do_evaluate(
    const vector_type& x, vector_type& residuals, matrix_type* jacobians)
{
    return guarded(last_error_,
        [&]
        {
            ensure_size(residuals, metadata_.num_residuals);
            residuals_(x, residuals);
            if (!residuals.allFinite())
            {
                return evaluation_status::invalid_trial;
            }
            if (jacobians != nullptr)
            {
                finite_difference_jacobian(residuals_,
                    x,
                    residuals,
                    relative_step_,
                    bounds_,
                    *jacobians,
                    counters_.residual_evaluations,
                    &workspace_);
                if (!jacobians->allFinite())
                {
                    return evaluation_status::invalid_trial;
                }
            }
            return evaluation_status::ok;
        });
}

evaluation_status finite_difference_residual_evaluator::do_jacobian(
    const vector_type& x, const vector_type& residuals_at_x, matrix_type& jacobian)
{
    return guarded(last_error_,
        [&]
        {
            finite_difference_jacobian(residuals_,
                x,
                residuals_at_x,
                relative_step_,
                bounds_,
                jacobian,
                counters_.residual_evaluations,
                &workspace_);
            return jacobian.allFinite() ? evaluation_status::ok : evaluation_status::invalid_trial;
        });
}

// -- delegating_residual_evaluator -------------------------------------------------
delegating_residual_evaluator::delegating_residual_evaluator(residual_function residuals,
    std::unique_ptr<residual_evaluator>                                        derivative_source,
    api::derivative_mode                                                       source)
    : residuals_(std::move(residuals)), inner_(std::move(derivative_source)),
      metadata_{inner_->metadata().num_parameters,
          inner_->metadata().num_residuals,
          source,
          inner_->metadata().supports_ceres}
{
}

std::optional<std::string> delegating_residual_evaluator::last_error() const
{
    return last_error_ ? last_error_ : inner_->last_error();
}

evaluation_status delegating_residual_evaluator::do_evaluate(
    const vector_type& x, vector_type& residuals, matrix_type* jacobians)
{
    if (jacobians == nullptr)
    {
        return guarded(last_error_,
            [&]
            {
                ensure_size(residuals, metadata_.num_residuals);
                residuals_(x, residuals);
                return residuals.allFinite() ? evaluation_status::ok
                                             : evaluation_status::invalid_trial;
            });
    }
    // The derivative source produces residuals as a by-product of its pass, so
    // one pass serves both outputs.
    const auto before = inner_->counters().residual_evaluations;
    const auto status = guarded(last_error_,
        [&]
        {
            ensure_size(residuals, metadata_.num_residuals);
            return inner_->evaluate(x, residuals, jacobians);
        });
    const auto passes = inner_->counters().residual_evaluations - before;
    counters_.residual_evaluations += passes > 0 ? passes - 1 : 0;
    return status;
}

evaluation_status delegating_residual_evaluator::do_jacobian(
    const vector_type& x, const vector_type& residuals_at_x, matrix_type& jacobian)
{
    const auto before = inner_->counters().residual_evaluations;
    const auto status =
        guarded(last_error_, [&] { return inner_->jacobian(x, residuals_at_x, jacobian); });
    counters_.residual_evaluations += inner_->counters().residual_evaluations - before;
    return status;
}

// -- provider adapter -----------------------------------------------------------------
namespace
{
class provider_adapter_evaluator final : public residual_evaluator
{
public:
    explicit provider_adapter_evaluator(std::shared_ptr<JacobianProvider> provider)
        : provider_(std::move(provider)),
          metadata_{
              provider_->num_parameters(), provider_->num_residuals(), provider_->source(), false}
    {
    }

    const provider_metadata&   metadata() const override { return metadata_; }
    std::optional<std::string> last_error() const override { return last_error_; }

protected:
    evaluation_status do_evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians) override
    {
        return guarded(last_error_,
            [&]
            {
                ensure_size(residuals, metadata_.num_residuals);
                if (jacobians != nullptr)
                {
                    ensure_size(*jacobians, metadata_.num_residuals, metadata_.num_parameters);
                    provider_->compute(x, residuals, *jacobians);
                    if (!jacobians->allFinite())
                    {
                        return evaluation_status::invalid_trial;
                    }
                }
                else
                {
                    provider_->residuals_only(x, residuals);
                }
                return residuals.allFinite() ? evaluation_status::ok
                                             : evaluation_status::invalid_trial;
            });
    }

    evaluation_status do_jacobian(
        const vector_type& x, const vector_type& /*residuals_at_x*/, matrix_type& jacobian) override
    {
        return guarded(last_error_,
            [&]
            {
                ensure_size(jacobian, metadata_.num_residuals, metadata_.num_parameters);
                provider_->jacobian_only(x, jacobian);
                counters_.residual_evaluations += provider_->residual_passes_per_jacobian();
                return jacobian.allFinite() ? evaluation_status::ok
                                            : evaluation_status::invalid_trial;
            });
    }

private:
    std::shared_ptr<JacobianProvider> provider_;
    provider_metadata                 metadata_;
    std::optional<std::string>        last_error_;
};
}  // namespace

std::unique_ptr<residual_evaluator> make_provider_evaluator(
    std::shared_ptr<JacobianProvider> provider)
{
    return std::make_unique<provider_adapter_evaluator>(std::move(provider));
}

// -- gradient evaluators --------------------------------------------------------------
callback_gradient_evaluator::callback_gradient_evaluator(std::size_t n,
    objective_function                                               objective,
    gradient_function                                                gradient,
    api::derivative_mode                                             source)
    : n_(n), objective_(std::move(objective)), gradient_(std::move(gradient)), source_(source)
{
}

evaluation_status callback_gradient_evaluator::do_evaluate(
    const vector_type& x, double& value, vector_type* gradient)
{
    return guarded(last_error_,
        [&]
        {
            value = objective_(x);
            if (gradient != nullptr)
            {
                if (!gradient_)
                {
                    last_error_ = "no gradient callback available";
                    return evaluation_status::fatal_error;
                }
                ensure_size(*gradient, n_);
                gradient_(x, *gradient);
                if (!gradient->allFinite())
                {
                    return evaluation_status::invalid_trial;
                }
            }
            return std::isfinite(value) ? evaluation_status::ok : evaluation_status::invalid_trial;
        });
}

evaluation_status callback_gradient_evaluator::do_gradient(
    const vector_type& x, vector_type& gradient)
{
    return guarded(last_error_,
        [&]
        {
            if (!gradient_)
            {
                last_error_ = "no gradient callback available";
                return evaluation_status::fatal_error;
            }
            ensure_size(gradient, n_);
            gradient_(x, gradient);
            return gradient.allFinite() ? evaluation_status::ok : evaluation_status::invalid_trial;
        });
}

finite_difference_gradient_evaluator::finite_difference_gradient_evaluator(
    std::size_t n, objective_function objective, double relative_step, api::bounds bounds)
    : n_(n), objective_(std::move(objective)), relative_step_(relative_step),
      bounds_(std::move(bounds))
{
}

evaluation_status finite_difference_gradient_evaluator::do_evaluate(
    const vector_type& x, double& value, vector_type* gradient)
{
    return guarded(last_error_,
        [&]
        {
            value = objective_(x);
            if (!std::isfinite(value))
            {
                return evaluation_status::invalid_trial;
            }
            if (gradient != nullptr)
            {
                finite_difference_gradient(objective_,
                    x,
                    &value,
                    relative_step_,
                    bounds_,
                    *gradient,
                    counters_.objective_evaluations);
                if (!gradient->allFinite())
                {
                    return evaluation_status::invalid_trial;
                }
            }
            return evaluation_status::ok;
        });
}

evaluation_status finite_difference_gradient_evaluator::do_gradient(
    const vector_type& x, vector_type& gradient)
{
    return guarded(last_error_,
        [&]
        {
            finite_difference_gradient(objective_,
                x,
                nullptr,
                relative_step_,
                bounds_,
                gradient,
                counters_.objective_evaluations);
            return gradient.allFinite() ? evaluation_status::ok : evaluation_status::invalid_trial;
        });
}

// -- the derivative resolvers ----------------------------------------------------------
namespace
{
solver_result rejected(solver_status status, std::string message, const vector_type& x)
{
    solver_result result;
    result.status     = status;
    result.parameters = x;
    result.message    = std::move(message);
    return result;
}

jacobian_function provider_callback(const std::shared_ptr<JacobianProvider>& provider)
{
    // Jacobian only: a callback Jacobian never re-runs the residual function.
    return [provider](const vector_type& x, matrix_type& jacobian)
    { provider->jacobian_only(x, jacobian); };
}

jacobian_function fd_callback(const residual_function& residuals, std::size_t m, double step)
{
    return [residuals, m, step](const vector_type& x, matrix_type& jacobian)
    {
        vector_type r = make_vector(m);
        residuals(x, r);
        std::size_t evaluations = 0;
        finite_difference_jacobian(residuals, x, r, step, api::bounds{}, jacobian, evaluations);
    };
}
}  // namespace

std::variant<resolved_derivatives, solver_result> resolve_derivatives(
    const least_squares_problem& problem, const solve_options& options, const vector_type& x)
{
    const std::size_t n = problem.num_parameters;
    const std::size_t m = problem.num_residuals;

    // The problem has one derivative slot; callbacks, AD and curve derivatives
    // are all providers that advertise their source().
    const auto provider  = problem.derivative_provider();
    const auto source_of = [&]
    { return provider ? provider->source() : derivative_mode::finite_difference; };

    // 1. Pick a concrete source, validating explicit requests strictly.
    derivative_mode source = options.derivatives;
    switch (options.derivatives)
    {
    case derivative_mode::supplied:
        if (!provider || provider->source() != derivative_mode::supplied)
        {
            return rejected(
                solver_status::invalid_problem, "supplied Jacobian required but not provided", x);
        }
        break;
    case derivative_mode::automatic_differentiation:
        if (!provider || provider->source() != derivative_mode::automatic_differentiation)
        {
            return rejected(solver_status::unsupported_capability,
                "automatic differentiation required but no compatible AD provider available",
                x);
        }
        break;
    case derivative_mode::finite_difference:
        break;
    case derivative_mode::automatic:
        // The provider's own source, else finite differences.
        source = source_of();
        break;
    }

    // 2. Build the evaluator and the plain callback for that source.
    resolved_derivatives resolved;
    resolved.source         = source;
    const bool use_provider = provider && provider->source() == source &&
                              (source != derivative_mode::finite_difference ||
                                  options.derivatives == derivative_mode::automatic);
    if (use_provider)
    {
        resolved.ad_factory = provider->ceres_factory();
        std::unique_ptr<residual_evaluator> inner =
            (source == derivative_mode::automatic_differentiation && resolved.ad_factory)
                ? resolved.ad_factory->create_evaluator()
                : make_provider_evaluator(provider);
        resolved.evaluator = std::make_unique<delegating_residual_evaluator>(
            problem.residuals, std::move(inner), source);
        resolved.jacobian_callback = provider_callback(provider);
    }
    else
    {
        // Explicit finite differences (or no provider): the one shared stencil.
        resolved.evaluator = std::make_unique<finite_difference_residual_evaluator>(
            n, m, problem.residuals, default_fd_step, problem.bounds);
        resolved.jacobian_callback = fd_callback(problem.residuals, m, default_fd_step);
    }
    return resolved;
}

std::variant<resolved_gradient, solver_result> resolve_gradient(
    const optimization_problem& problem, const solve_options& options, const vector_type& x)
{
    const std::size_t n        = problem.num_parameters;
    const auto        provider = problem.derivative_provider();

    derivative_mode source = options.derivatives;
    switch (options.derivatives)
    {
    case derivative_mode::supplied:
        if (!provider || provider->source() != derivative_mode::supplied)
        {
            return rejected(
                solver_status::invalid_problem, "supplied gradient required but not provided", x);
        }
        break;
    case derivative_mode::automatic_differentiation:
        if (!provider || provider->source() != derivative_mode::automatic_differentiation)
        {
            return rejected(solver_status::unsupported_capability,
                "automatic differentiation required but no compatible AD gradient provider "
                "available",
                x);
        }
        break;
    case derivative_mode::finite_difference:
        break;
    case derivative_mode::automatic:
        if (!provider)
        {
            return rejected(solver_status::unsupported_capability,
                "no gradient source: provide a gradient callback or gradient provider, or request "
                "derivative_mode::finite_difference",
                x);
        }
        source = provider->source();
        break;
    }

    resolved_gradient resolved;
    resolved.source = source;
    if (source == derivative_mode::finite_difference)
    {
        resolved.evaluator = std::make_unique<finite_difference_gradient_evaluator>(
            n, problem.objective, default_fd_step, problem.bounds);
    }
    else
    {
        gradient_function gradient = [provider](const vector_type& point, vector_type& g)
        { provider->compute(point, g); };
        resolved.evaluator = std::make_unique<callback_gradient_evaluator>(
            n, problem.objective, std::move(gradient), source);
    }

    auto* evaluator    = resolved.evaluator.get();
    resolved.objective = [evaluator](const vector_type& point)
    {
        double value = 0.0;
        if (evaluator->evaluate(point, value) == evaluation_status::fatal_error)
        {
            throw std::runtime_error(
                evaluator->last_error().value_or("objective evaluation failed"));
        }
        return value;
    };
    resolved.gradient = [evaluator](const vector_type& point, vector_type& g)
    {
        if (evaluator->gradient(point, g) == evaluation_status::fatal_error)
        {
            throw std::runtime_error(
                evaluator->last_error().value_or("gradient evaluation failed"));
        }
    };
    return resolved;
}
}  // namespace solverslib::api::detail
