#include "solvers/api/detail/routing.h"

#include <array>
#include <cmath>

#include "solvers/api/solve.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/petsc_tao_solver.h"

namespace solverslib::api::detail
{
namespace
{
using B = api::backend;
using A = api::algorithm;
using K = problem_kind;
using C = capability;
using R = requirement;

// One row per realizable (backend, algorithm, kind) triple. Adding a backend or
// algorithm means adding rows here; selection, validation and messages follow.
//
// Priority guide (higher wins among built, eligible rows):
//   60 derivative-free specialists (POUNDERS), 50 native default,
//   40 Ipopt, 35 Ceres for bounded problems, 30 TAO, <=20 pin-only methods.
constexpr std::array<route, 11> routes = {{
    // -- least squares ------------------------------------------------------
    {B::native,
        A::levenberg_marquardt,
        K::least_squares,
        C::derivative_free,
        R::always,
        50,
        false,
        false,
        false},
    {B::native,
        A::gauss_newton,
        K::least_squares,
        C::derivative_free,
        R::always,
        20,
        false,
        false,
        false},
    {B::native, A::lbfgs, K::least_squares, C::derivative_free, R::always, 10, false, false, false},
    // RNC-LM brings its own curve derivatives, so it needs no Jacobian source.
    {B::native,
        A::riemann_normal_coordinate_lm,
        K::least_squares,
        C::derivative_free,
        R::always,
        5,
        false,
        false,
        false},
    {B::ceres,
        A::levenberg_marquardt,
        K::least_squares,
        C::bounds | C::derivative_free | C::native_ad,
        R::ceres,
        35,
        false,
        false,
        false},
    {B::pounders,
        A::pounders,
        K::least_squares,
        C::bounds | C::derivative_free,
        R::petsc_tao,
        60,
        false,
        true,
        false},
    {B::petsc_tao,
        A::newton_krylov,
        K::least_squares,
        C::bounds,
        R::petsc_tao,
        30,
        true,
        false,
        true},
    {B::petsc_tao,
        A::pounders,
        K::least_squares,
        C::bounds | C::derivative_free,
        R::petsc_tao,
        5,
        false,
        true,
        false},
    // -- general objective --------------------------------------------------
    {B::native, A::lbfgs, K::objective, C::derivative_free, R::always, 50, false, false, false},
    {B::ipopt,
        A::interior_point,
        K::objective,
        C::bounds | C::nonlinear_constraints,
        R::ipopt,
        40,
        false,
        false,
        false},
    {B::petsc_tao,
        A::newton_krylov,
        K::objective,
        C::bounds | C::hessian_vector,
        R::petsc_tao,
        30,
        true,
        false,
        false},
}};

bool valid_lm_options(const lm_options& o)
{
    const auto positive    = [](double v) { return std::isfinite(v) && v > 0.0; };
    const auto nonnegative = [](double v) { return std::isfinite(v) && v >= 0.0; };
    const auto factor      = [](double v) { return std::isfinite(v) && v > 1.0; };
    return positive(o.initial_damping) && positive(o.geodesic_acceleration_step) &&
           positive(o.geodesic_acceleration_threshold) && positive(o.bold_acceptance_exponent) &&
           factor(o.initial_rejection_multiplier) && factor(o.damping_decrease_factor) &&
           factor(o.damping_increase_factor) && positive(o.damping_floor) &&
           positive(o.nielsen_damping_floor) && positive(o.damping_ceiling) &&
           positive(o.levenberg_marquardt_damping_ceiling) && positive(o.diagonal_scaling_floor) &&
           nonnegative(o.roundoff_noise_factor) && o.damping_floor < o.damping_ceiling &&
           o.nielsen_damping_floor < o.damping_ceiling &&
           o.damping_floor < o.levenberg_marquardt_damping_ceiling;
}

std::string describe(api::algorithm value, api::backend chosen)
{
    std::string text;
    if (value != A::automatic)
    {
        text += std::string("algorithm '") + to_string(value) + "'";
    }
    if (chosen != B::automatic)
    {
        text += (text.empty() ? "" : " on ") + std::string("backend '") + to_string(chosen) + "'";
    }
    return text.empty() ? "the request" : text;
}

}  // namespace

bool backend_availability::operator()(requirement need) const noexcept
{
    switch (need)
    {
    case requirement::always:
        return true;
    case requirement::ceres:
        return ceres;
    case requirement::petsc_tao:
        return petsc_tao;
    case requirement::ipopt:
        return ipopt;
    }
    return false;
}

backend_availability backend_availability::current()
{
    return {ceres_solver::is_supported(),
        petsc_tao_solver::is_supported(),
        ipopt_solver::is_supported()};
}

backend_availability backend_availability::all()
{
    return {true, true, true};
}

backend_availability backend_availability::none()
{
    return {};
}

const char* to_string(capability single_capability)
{
    switch (single_capability)
    {
    case capability::none:
        return "none";
    case capability::bounds:
        return "bounds";
    case capability::nonlinear_constraints:
        return "nonlinear constraints";
    case capability::hessian_vector:
        return "Hessian-vector products";
    case capability::derivative_free:
        return "derivative-free operation";
    case capability::native_ad:
        return "native automatic differentiation";
    }
    return "unknown";
}

capability required_capabilities(const problem_traits& traits)
{
    capability required = capability::none;
    if (traits.has_bounds)
    {
        required = required | capability::bounds;
    }
    if (traits.has_nonlinear_constraints)
    {
        required = required | capability::nonlinear_constraints;
    }
    if (traits.has_hessian_vector_product)
    {
        required = required | capability::hessian_vector;
    }
    // Objective adapters report a missing gradient themselves; only residual
    // problems can fall back to finite differences inside the dispatcher.
    if (traits.is_least_squares && !traits.has_jacobian)
    {
        required = required | capability::derivative_free;
    }
    return required;
}

route_decision select_route(const problem_traits& traits,
    const solve_options&                          options,
    const backend_availability&                   availability)
{
    const problem_kind kind     = traits.is_least_squares ? K::least_squares : K::objective;
    const capability   required = required_capabilities(traits);
    const bool auto_mode  = options.backend == B::automatic && options.algorithm == A::automatic;
    const bool has_source = traits.is_least_squares ? traits.has_jacobian : traits.has_gradient;
    const bool large      = is_large_scale(traits, options.policy);
    const bool ad_wanted  = options.derivatives == derivative_mode::automatic ||
                           options.derivatives == derivative_mode::automatic_differentiation;

    route_decision decision;
    const route*   best       = nullptr;
    int            best_score = 0;

    // Fallback bookkeeping for the error message: the matching row that is
    // closest to serving the request.
    bool       any_match     = false;
    int        fewest_unmet  = 1 << 8;
    capability closest_unmet = capability::none;

    for (const route& row : routes)
    {
        if (row.kind != kind)
        {
            continue;
        }
        if (options.backend != B::automatic && row.backend != options.backend)
        {
            continue;
        }
        if (options.algorithm != A::automatic && row.algorithm != options.algorithm)
        {
            continue;
        }
        any_match = true;

        const std::uint16_t missing =
            static_cast<std::uint16_t>(required) & ~static_cast<std::uint16_t>(row.supports);
        if (missing != 0)
        {
            int count = 0;
            for (std::uint16_t bits = missing; bits != 0; bits &= (bits - 1))
            {
                ++count;
            }
            if (count < fewest_unmet)
            {
                fewest_unmet  = count;
                closest_unmet = static_cast<capability>(missing & (~missing + 1));
            }
            continue;
        }
        if (auto_mode && ((row.derivative_free_only && has_source) ||
                             (row.needs_derivative_source && !has_source)))
        {
            continue;
        }

        int score = row.priority;
        score += availability(row.needs) ? 0 : -1000;
        score += (row.large_scale && large) ? 200 : 0;
        score += (has_all(row.supports, C::native_ad) && traits.has_autodiff_provider && ad_wanted)
                     ? 100
                     : 0;
        if (best == nullptr || score > best_score)
        {
            best       = &row;
            best_score = score;
        }
    }

    if (best != nullptr)
    {
        decision.chosen = best;
        return decision;
    }

    decision.failure = solver_status::unsupported_capability;
    if (!any_match)
    {
        decision.message =
            std::string(describe(options.algorithm, options.backend)) + " is not implemented for " +
            (traits.is_least_squares ? "residual least squares" : "general objective problems");
    }
    else if (closest_unmet != capability::none)
    {
        decision.message = std::string("no solver for ") +
                           describe(options.algorithm, options.backend) + " supports " +
                           to_string(closest_unmet);
    }
    else
    {
        decision.message = "no solver is eligible for this problem and option combination";
    }
    return decision;
}

std::optional<validation_error> validate_options(const solve_options& options, const vector_type& x)
{
    if (options.max_iterations <= 0 || options.max_function_evaluations < 0 ||
        options.function_tolerance < 0.0 || options.gradient_tolerance < 0.0 ||
        options.parameter_tolerance < 0.0)
    {
        return validation_error{solver_status::invalid_problem,
            "iteration/evaluation budgets and tolerances must be non-negative; "
            "max_iterations must be positive"};
    }
    if (options.max_function_evaluations > 0)
    {
        return validation_error{solver_status::unsupported_capability,
            "max_function_evaluations is not yet enforceable across all native and external "
            "adapters"};
    }
    if (!x.allFinite())
    {
        return validation_error{
            solver_status::invalid_problem, "initial_guess must contain only finite values"};
    }
    if (options.lm && !valid_lm_options(*options.lm))
    {
        return validation_error{solver_status::invalid_problem,
            "invalid lm options: damping, scaling, step and factor values must be finite and "
            "positive, factors greater than one, and damping floors below the ceilings"};
    }
    return std::nullopt;
}

std::optional<std::string> validate_bounds(const api::bounds& bounds, std::size_t n)
{
    if ((bounds.has_lower() && bounds.lower.size() != n) ||
        (bounds.has_upper() && bounds.upper.size() != n))
    {
        return "bound vectors must be empty or match num_parameters";
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        const bool bad_lower = bounds.has_lower() && !std::isfinite(bounds.lower[i]);
        const bool bad_upper = bounds.has_upper() && !std::isfinite(bounds.upper[i]);
        const bool inverted =
            bounds.has_lower() && bounds.has_upper() && bounds.lower[i] > bounds.upper[i];
        if (bad_lower || bad_upper || inverted)
        {
            return "bounds must be finite and lower <= upper";
        }
    }
    return std::nullopt;
}
}  // namespace solverslib::api::detail
