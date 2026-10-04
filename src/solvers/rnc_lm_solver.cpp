#include "solvers/rnc_lm_solver.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

namespace solverslib
{
namespace
{
bool valid_options(const api::rnc_lm_options& cfg, const api::solve_options& options)
{
    const auto positive  = [](double v) { return std::isfinite(v) && v > 0.; };
    const auto tolerance = [](double v) { return std::isfinite(v) && v >= 0.; };
    return cfg.order >= 1 && cfg.order <= 4 && cfg.max_curve_trials > 0 &&
           positive(cfg.acceptance_threshold) && cfg.acceptance_threshold < 1. &&
           positive(cfg.contraction_min) && cfg.contraction_min < cfg.contraction_max &&
           cfg.contraction_max < 1. && positive(cfg.initial_damping) &&
           positive(cfg.damping_floor) && positive(cfg.damping_ceiling) &&
           cfg.damping_floor <= cfg.initial_damping && cfg.initial_damping <= cfg.damping_ceiling &&
           cfg.damping_floor < cfg.damping_ceiling && positive(cfg.diagonal_scaling_floor) &&
           options.max_iterations > 0 && tolerance(options.function_tolerance) &&
           tolerance(options.gradient_tolerance) && tolerance(options.parameter_tolerance);
}

bool valid_derivatives(const rnc_curve_derivatives& d, int order, index_type m, index_type n)
{
    if (d.residual.size() != static_cast<size_t>(order + 1) ||
        d.jacobian.size() != static_cast<size_t>(std::max(0, order - 2) + 1))
        return false;
    for (const auto& r : d.residual)
        if (r.size() != m || !r.allFinite())
            return false;
    for (const auto& j : d.jacobian)
        if (j.rows() != m || j.cols() != n || !j.allFinite())
            return false;
    return true;
}
}  // namespace

api::solver_result solve_rnc_lm(const api::least_squares_problem& problem,
    const vector_type&                                            initial_guess,
    const api::solve_options&                                     options)
{
    using api::solver_status;
    const auto         cfg = options.rnc_lm.value_or(api::rnc_lm_options{});
    api::solver_result result;
    result.parameters                  = initial_guess;
    result.algorithm                   = api::algorithm::riemann_normal_coordinate_lm;
    result.backend                     = api::backend::native;
    result.effective_derivative_source = problem.rnc_derivative_source;
    const auto finish                  = [&](solver_status status, const char* message)
    {
        result.status  = status;
        result.message = message;
        return result;
    };
    const auto n = static_cast<index_type>(problem.num_parameters);
    const auto m = static_cast<index_type>(problem.num_residuals);
    if (!valid_options(cfg, options) || n <= 0 || m <= 0 || initial_guess.size() != n ||
        !initial_guess.allFinite() || !problem.residuals)
        return finish(solver_status::invalid_problem, "Invalid RNC-LM problem or options");
    if (!problem.bounds.empty() || options.max_function_evaluations != 0)
        return finish(solver_status::unsupported_capability,
            "RNC-LM does not support bounds or evaluation budgets");
    if (!problem.rnc_derivatives)
        return finish(solver_status::unsupported_capability,
            "RNC-LM requires analytic or Taylor curve derivatives");
    if ((problem.rnc_derivative_source != api::derivative_mode::supplied &&
            problem.rnc_derivative_source != api::derivative_mode::automatic_differentiation) ||
        (options.derivatives != api::derivative_mode::automatic &&
            options.derivatives != problem.rnc_derivative_source))
        return finish(solver_status::unsupported_capability,
            "Requested derivative mode is incompatible with RNC curve derivatives");

    vector_type r(m), gradient(n);
    matrix_type j(m, n);
    const auto  refresh = [&]()
    {
        rnc_curve_derivatives d;
        ++result.jacobian_evaluations;
        problem.rnc_derivatives(result.parameters, {}, 1, d);
        if (!valid_derivatives(d, 1, m, n))
            return false;
        r        = d.residual[0];
        j        = d.jacobian[0];
        gradient = j.transpose() * r;
        ++result.gradient_evaluations;
        result.residual_norm = r.stableNorm();
        result.objective     = .5 * r.squaredNorm();
        result.gradient_norm = gradient.stableNorm();
        return std::isfinite(result.objective) && gradient.allFinite();
    };
    const auto converged = [&]()
    {
        return *result.residual_norm <= options.function_tolerance ||
               *result.gradient_norm <= options.gradient_tolerance;
    };
    double lambda = cfg.initial_damping;
    try
    {
        if (!refresh())
            return finish(solver_status::numerical_failure, "Invalid RNC base derivatives");
        if (converged())
            return finish(
                solver_status::converged, "RNC-LM residual or gradient tolerance reached");
        for (int iteration = 0; iteration < options.max_iterations; ++iteration)
        {
            ++result.iterations;  // Counts attempted curves, including rejected curves.
            matrix_type metric = j.transpose() * j;
            vector_type diagonal(n);
            for (index_type i = 0; i < n; ++i)
            {
                diagonal[i] = std::max(metric(i, i), cfg.diagonal_scaling_floor);
                metric(i, i) += lambda * diagonal[i];
            }
            bool                     accepted    = false;
            bool                     curve_valid = metric.allFinite();
            std::vector<vector_type> coefficients;
            double                   sigma       = 0.;
            double                   damped_norm = 0.;
            if (curve_valid)
            {
                positive_definite_solver factorization(metric);
                curve_valid = factorization.valid();
                if (curve_valid)
                {
                    const vector_type velocity = factorization.solve(-gradient);
                    sigma                      = -gradient.dot(velocity);
                    damped_norm                = velocity.dot(diagonal.cwiseProduct(velocity));
                    curve_valid = velocity.allFinite() && std::isfinite(sigma) && sigma > 0. &&
                                  std::isfinite(damped_norm);
                    coefficients.push_back(velocity);
                }
                // Liu & Zhang (2026), Eq. (21). All derivatives are evaluated
                // along theta_<order; every solve reuses the factorization.
                for (int order = 2; curve_valid && order <= cfg.order; ++order)
                {
                    rnc_curve_derivatives derivatives;
                    ++result.jacobian_evaluations;
                    problem.rnc_derivatives(result.parameters, coefficients, order, derivatives);
                    curve_valid = valid_derivatives(derivatives, order, m, n);
                    if (!curve_valid)
                        break;
                    vector_type defect   = vector_type::Zero(n);
                    double      binomial = 1.;
                    for (int k = 0; k <= order - 2; ++k)
                    {
                        defect += binomial * derivatives.jacobian[k].transpose() *
                                  derivatives.residual[order - k];
                        if (k < order - 2)
                            binomial *= double(order - 2 - k) / double(k + 1);
                    }
                    const vector_type coefficient = factorization.solve(-defect);
                    curve_valid                   = coefficient.allFinite();
                    coefficients.push_back(coefficient);
                }
            }
            double t = 1.;
            for (int trial = 0; curve_valid && trial < cfg.max_curve_trials; ++trial)
            {
                vector_type displacement = vector_type::Zero(n);
                double      weight       = 1.;
                for (size_t q = 0; q < coefficients.size(); ++q)
                {
                    weight *= t / double(q + 1);
                    displacement += weight * coefficients[q];
                }
                const vector_type candidate = result.parameters + displacement;
                vector_type       trial_residual(m);
                bool              finite = candidate.allFinite();
                if (finite)
                {
                    ++result.residual_evaluations;
                    problem.residuals(candidate, trial_residual);
                    finite = trial_residual.size() == m && trial_residual.allFinite();
                }
                const double trial_cost = finite ? .5 * trial_residual.squaredNorm()
                                                 : std::numeric_limits<double>::infinity();
                // Eq. (34): predict with t*v, not the corrected displacement.
                const double predicted =
                    (t - .5 * t * t) * sigma + .5 * lambda * t * t * damped_norm;
                const double actual = finite ? .5 * (r - trial_residual).dot(r + trial_residual)
                                             : -std::numeric_limits<double>::infinity();
                const double rho =
                    predicted > 0. ? actual / predicted : -std::numeric_limits<double>::infinity();
                if (std::isfinite(trial_cost) && std::isfinite(rho) &&
                    rho > cfg.acceptance_threshold)
                {
                    result.parameters    = candidate;
                    result.objective     = trial_cost;
                    result.residual_norm = trial_residual.stableNorm();
                    result.gradient_norm.reset();
                    result.step_norm = displacement.stableNorm();
                    ++result.accepted_steps;
                    accepted = true;
                    // Eq. (40), separate from Nielsen and bold acceptance.
                    if (rho < .25)
                        lambda = std::min(2. * lambda, cfg.damping_ceiling);
                    else if (rho > .75)
                        lambda = std::max(lambda / 3., cfg.damping_floor);
                    if (!refresh())
                        return finish(solver_status::numerical_failure,
                            "Invalid RNC derivatives at accepted point");
                    SOLVERS_LOG_IF(INFO,
                        options.verbose,
                        "RNC-LM iteration {} | order {} | t = {} | cost = {} | rho = {} | lambda = "
                        "{}",
                        result.iterations,
                        cfg.order,
                        t,
                        result.objective,
                        rho,
                        lambda);
                    if (converged())
                        return finish(solver_status::converged,
                            "RNC-LM residual or gradient tolerance reached");
                    if (*result.step_norm <=
                        options.parameter_tolerance *
                            (result.parameters.stableNorm() + options.parameter_tolerance))
                        return finish(
                            solver_status::converged, "RNC-LM parameter tolerance reached");
                    break;
                }
                ++result.rejected_steps;  // Rejected trial points, not outer curves.
                // Eqs. (37)-(38): scalar quadratic interpolation along the
                // fixed polynomial curve; no new Jacobian or coefficients.
                const double denominator = 2. * (trial_cost - result.objective + sigma * t);
                double       next_t      = std::isfinite(denominator) && denominator > 0.
                                               ? sigma * t * t / denominator
                                               : cfg.contraction_min * t;
                if (!std::isfinite(next_t))
                    next_t = cfg.contraction_min * t;
                t = std::clamp(next_t, cfg.contraction_min * t, cfg.contraction_max * t);
            }
            if (!accepted)
            {
                if (lambda >= cfg.damping_ceiling)
                    return finish(solver_status::numerical_failure,
                        "RNC-LM exhausted damping without an acceptable curve");
                lambda = std::min(2. * lambda, cfg.damping_ceiling);
            }
        }
    }
    catch (const std::exception& error)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("RNC-LM callback failed: ") + error.what();
        return result;
    }
    return finish(solver_status::max_iterations, "RNC-LM reached its curve iteration budget");
}
}  // namespace solverslib
