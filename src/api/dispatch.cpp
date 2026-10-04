#include "solvers/api/solve.h"

#include <cmath>
#include <exception>
#include <string>
#include <variant>
#include <vector>

#include "detail/backend_status.h"
#include "detail/native_result.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solvers/api/detail/evaluators.h"
#include "solvers/api/detail/routing.h"
#include "solvers/ceres_solver.h"

#if SOLVERS_HAS_CERES
#include <ceres/ceres.h>
#endif
#include "solvers/gauss_newton_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/lbfgs_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/petsc_tao_solver.h"
#include "solvers/rnc_lm_solver.h"

// Central dispatcher for the problem-structure API. It derives traits, resolves
// backend::automatic / algorithm::automatic, validates the request, and either
// runs the native kernels or reports why a backend cannot serve the request.
// Only the native backend is wired in this slice; every external backend is
// reported as unavailable rather than silently degraded (review sections 4-6,
// and F01 on rejecting rather than dropping unsupported capabilities).
namespace solverslib::api
{
namespace
{
solver_status translate_native(native_convergence status)
{
    switch (status)
    {
    case native_convergence::gradient_converged:
    case native_convergence::parameter_converged:
    case native_convergence::function_converged:
        return solver_status::converged;
    case native_convergence::not_converged:
        return solver_status::max_iterations;
    case native_convergence::numerical_failure:
        return solver_status::numerical_failure;
    case native_convergence::stalled:
        return solver_status::stalled;
    }
    return solver_status::numerical_failure;
}

levenberg_marquardt_solver_enum map_lm_variant(lm_variant value)
{
    switch (value)
    {
    case lm_variant::levenberg_marquardt:
        return levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT;
    case lm_variant::quadratic_interpolation:
        return levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION;
    case lm_variant::nielsen:
        return levenberg_marquardt_solver_enum::NIELSEN;
    }
    return levenberg_marquardt_solver_enum::NIELSEN;
}

void apply_lm_options(const lm_options& lm, solver_options_lm_builder& builder)
{
    builder.with_type(map_lm_variant(lm.variant))
        .with_linear_solver(lm.linear_solver == lm_linear_solver::augmented_qr
                                ? levenberg_marquardt_linear_solver_enum::AUGMENTED_QR
                                : levenberg_marquardt_linear_solver_enum::NORMAL_LDLT)
        .with_bold_acceptance(lm.bold_acceptance)
        .with_bold_acceptance_exponent(lm.bold_acceptance_exponent)
        .with_geodesic_acceleration(lm.geodesic_acceleration)
        .with_geodesic_acceleration_threshold(lm.geodesic_acceleration_threshold)
        .with_geodesic_acceleration_step(lm.geodesic_acceleration_step)
        .with_initial_damping(lm.initial_damping)
        .with_initial_rejection_multiplier(lm.initial_rejection_multiplier)
        .with_damping_decrease_factor(lm.damping_decrease_factor)
        .with_damping_increase_factor(lm.damping_increase_factor)
        .with_damping_floor(lm.damping_floor)
        .with_nielsen_damping_floor(lm.nielsen_damping_floor)
        .with_damping_ceiling(lm.damping_ceiling)
        .with_levenberg_marquardt_damping_ceiling(lm.levenberg_marquardt_damping_ceiling)
        .with_diagonal_scaling_floor(lm.diagonal_scaling_floor)
        .with_roundoff_noise_factor(lm.roundoff_noise_factor);
}

solver_status translate_outcome(backend_outcome outcome)
{
    switch (outcome)
    {
    case backend_outcome::converged:
        return solver_status::converged;
    case backend_outcome::budget_exhausted:
        return solver_status::max_iterations;
    case backend_outcome::stalled:
        return solver_status::stalled;
    case backend_outcome::infeasible:
        return solver_status::infeasible;
    case backend_outcome::user_stopped:
        return solver_status::user_stopped;
    case backend_outcome::failed:
        return solver_status::numerical_failure;
    }
    return solver_status::numerical_failure;
}

solver_result failed(solver_status status, const std::string& message, const vector_type& x)
{
    solver_result result;
    result.status     = status;
    result.parameters = x;
    result.message    = message;
    return result;
}

solver_result from_native(const native_result& out,
    const vector_type&                         x,
    api::algorithm                             alg,
    api::derivative_mode                       deriv_mode,
    const detail::evaluation_counters&         counters)
{
    solver_result result;
    result.status                      = translate_native(out.status);
    result.parameters                  = x;
    result.residual_norm               = out.residual_norm;
    result.objective                   = 0.5 * out.residual_norm * out.residual_norm;
    result.iterations                  = out.iterations;
    result.backend                     = backend::native;
    result.algorithm                   = alg;
    result.backend_status              = static_cast<int>(out.status);
    result.effective_derivative_source = deriv_mode;
    result.residual_evaluations        = counters.residual_evaluations;
    result.jacobian_evaluations        = counters.jacobian_evaluations;
    result.gradient_norm               = out.gradient_norm;
    result.step_norm                   = out.step_norm;
    result.accepted_steps              = out.accepted_steps;
    result.rejected_steps              = out.rejected_steps;
    switch (result.status)
    {
    case solver_status::converged:
        result.message = "native converged";
        break;
    case solver_status::numerical_failure:
        result.message = out.message.empty() ? "native solver failed" : out.message;
        break;
    case solver_status::stalled:
        result.message = "native solver stalled: " + out.message;
        break;
    default:
        result.message = "native reached iteration limit";
        break;
    }
    return result;
}
}  // namespace

problem_traits inspect(const least_squares_problem& problem)
{
    const auto provider = problem.derivative_provider();

    problem_traits traits;
    traits.is_least_squares      = true;
    traits.has_callable_jacobian = provider && provider->source() == derivative_mode::supplied;
    traits.has_jacobian_provider = !!provider;
    traits.has_autodiff_provider = provider && provider->ceres_factory() != nullptr;
    // has_jacobian is true when any derivative source is available
    traits.has_jacobian   = !!provider;
    traits.has_bounds     = !problem.bounds.empty();
    traits.num_parameters = problem.num_parameters;
    traits.num_residuals  = problem.num_residuals;
    return traits;
}

problem_traits inspect(const optimization_problem& problem)
{
    const auto provider = problem.derivative_provider();

    problem_traits traits;
    traits.is_least_squares           = false;
    traits.has_gradient_provider      = !!provider;
    traits.has_gradient               = !!provider;
    traits.has_hessian                = problem.hessian.has_value();
    traits.has_hessian_vector_product = problem.hessian_vector.has_value();
    traits.has_bounds                 = !problem.bounds.empty();
    traits.has_nonlinear_constraints  = !problem.constraints.empty();
    traits.num_parameters             = problem.num_parameters;
    return traits;
}

bool is_large_scale(const problem_traits& traits, const dispatch_policy& policy)
{
    if (policy.prefer_matrix_free && traits.has_hessian_vector_product)
    {
        return true;
    }
    return traits.num_parameters > policy.large_parameter_threshold ||
           traits.num_residuals > policy.large_residual_threshold;
}

algorithm select_algorithm(const problem_traits& traits, const solve_options& options)
{
    const auto decision = detail::select_route(traits, options);
    return decision.chosen != nullptr ? decision.chosen->algorithm : options.algorithm;
}

backend select_backend(const problem_traits& traits, const solve_options& options)
{
    const auto decision = detail::select_route(traits, options);
    return decision.chosen != nullptr ? decision.chosen->backend : options.backend;
}

namespace
{
// Shared native least-squares execution once validation has passed.
solver_result run_native_least_squares(const least_squares_problem& problem,
    const vector_type&                                              initial_guess,
    const solve_options&                                            options,
    algorithm                                                       alg)
{
    if (alg == algorithm::riemann_normal_coordinate_lm)
    {
        return solve_rnc_lm(problem, initial_guess, options);
    }

    auto resolution = detail::resolve_derivatives(problem, options, initial_guess);
    if (auto* error = std::get_if<solver_result>(&resolution))
    {
        error->backend   = backend::native;
        error->algorithm = alg;
        return std::move(*error);
    }
    auto& resolved = std::get<detail::resolved_derivatives>(resolution);

    vector_type x = initial_guess;
    try
    {
        auto& evaluator = *resolved.evaluator;
        switch (alg)
        {
        case algorithm::gauss_newton:
        {
            auto native_opts = solver_options_gn_builder()
                                   .with_max_iterations(options.max_iterations)
                                   .with_function_tolerance(options.function_tolerance)
                                   .with_gradient_tolerance(options.gradient_tolerance)
                                   .with_parameter_tolerance(options.parameter_tolerance)
                                   .with_verbose(options.verbose)
                                   .build();
            gauss_newton_solver solver(evaluator);
            const auto          out = solver.solve(x, *native_opts);
            return from_native(out, x, alg, resolved.source, evaluator.counters());
        }
        case algorithm::lbfgs:
        {
            auto native_opts = solver_options_bfgs_builder()
                                   .with_max_iterations(options.max_iterations)
                                   .with_function_tolerance(options.function_tolerance)
                                   .with_gradient_tolerance(options.gradient_tolerance)
                                   .with_parameter_tolerance(options.parameter_tolerance)
                                   .with_verbose(options.verbose)
                                   .build();
            lbfgs_solver solver(evaluator);
            const auto   out = solver.solve(x, *native_opts);
            return from_native(out, x, algorithm::lbfgs, resolved.source, evaluator.counters());
        }
        case algorithm::levenberg_marquardt:
        default:
        {
            auto builder = solver_options_lm_builder()
                               .with_max_iterations(options.max_iterations)
                               .with_function_tolerance(options.function_tolerance)
                               .with_gradient_tolerance(options.gradient_tolerance)
                               .with_parameter_tolerance(options.parameter_tolerance)
                               .with_verbose(options.verbose);
            if (options.lm)
            {
                apply_lm_options(*options.lm, builder);
            }
            auto                       native_opts = builder.build();
            levenberg_marquardt_solver solver(evaluator);
            const auto                 out = solver.solve(x, *native_opts);
            return from_native(
                out, x, algorithm::levenberg_marquardt, resolved.source, evaluator.counters());
        }
        }
    }
    catch (const std::exception& e)
    {
        // Kernels report evaluation failures through native_result; anything
        // else (a failed line search, an invalid option) must not escape the API.
        auto result                        = failed(solver_status::numerical_failure,
            std::string("native solver threw: ") + e.what(),
            initial_guess);
        result.backend                     = backend::native;
        result.algorithm                   = alg;
        result.effective_derivative_source = resolved.source;
        return result;
    }
}

// Residual L2 norm at a point, used to fill result diagnostics for the external
// adapters (whose bool/void return does not carry it).
double residual_norm_at(const least_squares_problem& problem, const vector_type& x)
{
    vector_type r = make_vector(problem.num_residuals);
    problem.residuals(x, r);
    return r.norm();
}

// -- Ceres option mapping ----------------------------------------------------
linear_solver_enum map_ceres_linear_solver(ceres_linear_solver value)
{
    switch (value)
    {
    case ceres_linear_solver::dense_qr:
        return linear_solver_enum::DENSE_QR;
    case ceres_linear_solver::dense_normal_cholesky:
        return linear_solver_enum::DENSE_NORMAL_CHOLESKY;
    case ceres_linear_solver::sparse_normal_cholesky:
        return linear_solver_enum::SPARSE_NORMAL_CHOLESKY;
    case ceres_linear_solver::dense_schur:
        return linear_solver_enum::DENSE_SCHUR;
    case ceres_linear_solver::sparse_schur:
        return linear_solver_enum::SPARSE_SCHUR;
    case ceres_linear_solver::iterative_schur:
        return linear_solver_enum::ITERATIVE_SCHUR;
    case ceres_linear_solver::cgnr:
        return linear_solver_enum::CGNR;
    }
    return linear_solver_enum::DENSE_QR;
}

trust_region_strategy_enum map_ceres_trust_region(ceres_trust_region_strategy value)
{
    switch (value)
    {
    case ceres_trust_region_strategy::levenberg_marquardt:
        return trust_region_strategy_enum::LEVENBERG_MARQUARDT;
    case ceres_trust_region_strategy::dogleg:
        return trust_region_strategy_enum::DOGLEG;
    }
    return trust_region_strategy_enum::LEVENBERG_MARQUARDT;
}

solver_result run_ceres(const least_squares_problem& problem,
    const vector_type&                               initial_guess,
    const solve_options&                             options)
{
    solver_result result;
    result.parameters = initial_guess;
    result.backend    = backend::ceres;
    result.algorithm  = options.algorithm == algorithm::automatic ? algorithm::levenberg_marquardt
                                                                  : options.algorithm;

    if (!ceres_solver::is_supported())
    {
        result.status  = solver_status::backend_unavailable;
        result.message = "Ceres backend was not compiled in (SOLVERS_ENABLE_CERES=OFF)";
        return result;
    }

    auto resolution = detail::resolve_derivatives(problem, options, initial_guess);
    if (auto* error = std::get_if<solver_result>(&resolution))
    {
        error->backend   = backend::ceres;
        error->algorithm = result.algorithm;
        return std::move(*error);
    }
    const auto& resolved             = std::get<detail::resolved_derivatives>(resolution);
    const auto resolved_derivatives = resolved.source;

    // Default to DENSE_QR: a native-parity dense least-squares path rather than
    // Ceres' sparse default, unless the caller asked otherwise.
    auto builder = solver_options_ceres_builder()
                       .with_max_iterations(options.max_iterations)
                       .with_function_tolerance(options.function_tolerance)
                       .with_gradient_tolerance(options.gradient_tolerance)
                       .with_parameter_tolerance(options.parameter_tolerance)
                       .with_verbose(options.verbose)
                       .with_linear_solver_type(linear_solver_enum::DENSE_QR);
    if (options.ceres)
    {
        builder.with_linear_solver_type(map_ceres_linear_solver(options.ceres->linear_solver))
            .with_trust_region_strategy_type(
                map_ceres_trust_region(options.ceres->trust_region_strategy))
            .with_num_threads(options.ceres->num_threads)
            .with_max_solver_time_in_seconds(options.ceres->max_solver_time_seconds);
    }
    auto ceres_opts = builder.build();

    std::vector<double> parameters(
        initial_guess.data(), initial_guess.data() + initial_guess.size());

    // Ceres-native AD takes the factory; an analytic or provider Jacobian goes
    // in as a callback; finite differences are left to Ceres' own numeric
    // differentiation.
    std::shared_ptr<const api::detail::provider_factory> provider_to_use;
    jacobian_function                                    jacobian_to_use;
    if (resolved.ad_factory)
    {
        provider_to_use = resolved.ad_factory;
    }
    else if (resolved_derivatives != derivative_mode::finite_difference)
    {
        jacobian_to_use = resolved.jacobian_callback;
    }

    ceres_solver solver = (jacobian_to_use && !provider_to_use)
                              ? ceres_solver(problem.num_parameters,
                                    problem.num_residuals,
                                    problem.residuals,
                                    jacobian_to_use,
                                    problem.bounds.lower,
                                    problem.bounds.upper)
                              : ceres_solver(problem.num_parameters,
                                    problem.num_residuals,
                                    problem.residuals,
                                    provider_to_use,
                                    problem.bounds.lower,
                                    problem.bounds.upper);

#if SOLVERS_HAS_CERES
    ceres::Solver::Summary summary;
    try
    {
        solver.solve_with_summary(parameters, *ceres_opts, &summary);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("Ceres threw: ") + e.what();
        return result;
    }

    // Extract truthful results from Ceres summary
    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    const double rnorm   = residual_norm_at(problem, result.parameters);
    result.residual_norm = rnorm;
    result.objective     = 0.5 * rnorm * rnorm;

    // Map Ceres termination to solver status truthfully; the raw type is kept.
    result.backend_status = static_cast<int>(summary.termination_type);
    switch (summary.termination_type)
    {
    case ceres::TerminationType::CONVERGENCE:
        result.status  = solver_status::converged;
        result.message = "Ceres converged";
        break;
    case ceres::TerminationType::NO_CONVERGENCE:
        // Budget exhausted: the iterate is the best point found, so it is usable.
        if (summary.IsSolutionUsable() && !summary.iterations.empty())
        {
            result.status  = solver_status::max_iterations;
            result.message = "Ceres reached its iteration or time limit (solution usable)";
        }
        else
        {
            result.status  = solver_status::numerical_failure;
            result.message = "Ceres did not converge";
        }
        break;
    case ceres::TerminationType::USER_SUCCESS:
        result.status  = solver_status::user_stopped;
        result.message = "Ceres stopped by a user callback that reported success";
        break;
    case ceres::TerminationType::USER_FAILURE:
        result.status  = solver_status::user_stopped;
        result.message = "Ceres stopped by a user callback that reported failure";
        break;
    case ceres::TerminationType::FAILURE:
    default:
        result.status  = solver_status::numerical_failure;
        result.message = "Ceres failed: " + summary.message;
        break;
    }

    result.residual_evaluations = static_cast<std::size_t>(summary.num_residual_evaluations);
    result.jacobian_evaluations = static_cast<std::size_t>(summary.num_jacobian_evaluations);
    result.accepted_steps       = static_cast<std::size_t>(summary.num_successful_steps);
    result.rejected_steps       = static_cast<std::size_t>(summary.num_unsuccessful_steps);
    if (!summary.iterations.empty())
    {
        result.gradient_norm = summary.iterations.back().gradient_norm;
        result.step_norm     = summary.iterations.back().step_norm;
    }

    result.iterations                  = static_cast<int>(summary.iterations.size());
    result.effective_derivative_source = resolved_derivatives;
    return result;
#else
    (void)solver;
    result.status  = solver_status::backend_unavailable;
    result.message = "Ceres backend was not compiled in (SOLVERS_ENABLE_CERES=OFF)";
    return result;
#endif
}

// -- PETSc/TAO option mapping ------------------------------------------------
tao_algorithm_enum map_tao_algorithm(tao_algorithm value, bool least_squares)
{
    switch (value)
    {
    case tao_algorithm::pounders:
        return tao_algorithm_enum::POUNDERS;
    case tao_algorithm::brgn:
        return tao_algorithm_enum::BRGN;
    case tao_algorithm::nls:
        return tao_algorithm_enum::NLS;
    case tao_algorithm::ntr:
        return tao_algorithm_enum::NTR;
    case tao_algorithm::ntl:
        return tao_algorithm_enum::NTL;
    case tao_algorithm::lmvm:
        return tao_algorithm_enum::LMVM;
    case tao_algorithm::bqnls:
        return tao_algorithm_enum::BQNLS;
    case tao_algorithm::bnls:
        return tao_algorithm_enum::BNLS;
    case tao_algorithm::automatic:
        return least_squares ? tao_algorithm_enum::POUNDERS : tao_algorithm_enum::LMVM;
    }
    return least_squares ? tao_algorithm_enum::POUNDERS : tao_algorithm_enum::LMVM;
}

// TAO least-squares path, serving both backend::pounders and the large-scale
// backend::petsc_tao least-squares route.
solver_result run_petsc_tao_least_squares(const least_squares_problem& problem,
    const vector_type&                                                 initial_guess,
    const solve_options&                                               options,
    backend                                                            chosen,
    algorithm                                                          alg)
{
    const petsc_tao_options cfg = options.petsc_tao.value_or(petsc_tao_options{});

    solver_result result;
    result.parameters = initial_guess;
    result.backend    = chosen;
    result.algorithm  = alg;

    if (!petsc_tao_solver::is_supported())
    {
        result.status  = solver_status::backend_unavailable;
        result.message = "PETSc/TAO backend was not compiled in (SOLVERS_ENABLE_PETSC=OFF)";
        return result;
    }

    auto resolution = detail::resolve_derivatives(problem, options, initial_guess);
    if (auto* error = std::get_if<solver_result>(&resolution))
    {
        error->backend   = chosen;
        error->algorithm = alg;
        return std::move(*error);
    }
    const auto& resolved             = std::get<detail::resolved_derivatives>(resolution);
    const auto resolved_derivatives = resolved.source;

    // POUNDERS by default on the pounders route; BRGN default on the general
    // TAO route (Gauss-Newton for large residuals with a Jacobian).
    tao_algorithm requested = cfg.algorithm;
    if (requested == tao_algorithm::automatic)
    {
        requested = alg == algorithm::pounders ? tao_algorithm::pounders : tao_algorithm::brgn;
    }

    auto petsc_opts = solver_options_petsc_builder()
                          .with_tao_type(map_tao_algorithm(requested, /*least_squares=*/true))
                          .with_max_iterations(options.max_iterations)
                          .with_gatol(cfg.gatol)
                          .with_grtol(cfg.grtol)
                          .with_verbose(options.verbose)
                          .build();

    std::vector<double> parameters(
        initial_guess.data(), initial_guess.data() + initial_guess.size());

    // POUNDERS is derivative-free: it only gets a Jacobian that already exists.
    // The Gauss-Newton style TAO algorithms get one always (numeric if needed).
    petsc_tao_solver::jacobian_type jac;
    if (resolved.jacobian_callback &&
        !(requested == tao_algorithm::pounders &&
            resolved_derivatives == derivative_mode::finite_difference))
    {
        jac = resolved.jacobian_callback;
    }

    petsc_tao_solver solver(problem.num_parameters,
        problem.num_residuals,
        problem.residuals,
        jac,
        problem.bounds.lower,
        problem.bounds.upper);

    backend_solve_status outcome;
    try
    {
        outcome = solver.solve_with_status(parameters, *petsc_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("PETSc/TAO threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    const double rnorm                 = residual_norm_at(problem, result.parameters);
    result.residual_norm               = rnorm;
    result.objective                   = 0.5 * rnorm * rnorm;
    result.status                      = translate_outcome(outcome.outcome);
    result.backend_status              = outcome.native_code;
    result.effective_derivative_source = resolved_derivatives;
    if (outcome.iterations)
    {
        result.iterations = *outcome.iterations;
    }
    result.message = outcome.message;
    return result;
}

// Native scalar-objective path via L-BFGS.
solver_result run_native_optimization(const optimization_problem& problem,
    const vector_type&                                            initial_guess,
    const solve_options&                                          options,
    algorithm                                                     alg)
{
    auto resolution = detail::resolve_gradient(problem, options, initial_guess);
    if (auto* error = std::get_if<solver_result>(&resolution))
    {
        error->backend   = backend::native;
        error->algorithm = alg;
        return std::move(*error);
    }
    auto& resolved  = std::get<detail::resolved_gradient>(resolution);
    auto& evaluator = *resolved.evaluator;

    vector_type x = initial_guess;

    auto native_opts = solver_options_bfgs_builder()
                           .with_max_iterations(options.max_iterations)
                           .with_function_tolerance(options.function_tolerance)
                           .with_gradient_tolerance(options.gradient_tolerance)
                           .with_parameter_tolerance(options.parameter_tolerance)
                           .with_verbose(options.verbose)
                           .build();

    solver_result result;
    result.parameters                  = x;
    result.backend                     = backend::native;
    result.algorithm                   = algorithm::lbfgs;
    result.effective_derivative_source = resolved.source;
    try
    {
        lbfgs_solver solver(evaluator);
        const auto   out      = solver.solve(x, *native_opts);
        result.status         = translate_native(out.status);
        result.iterations     = out.iterations;
        result.parameters     = x;
        result.objective      = resolved.objective(x);
        result.gradient_norm  = out.gradient_norm;
        result.step_norm      = out.step_norm;
        result.accepted_steps = out.accepted_steps;
        result.rejected_steps = out.rejected_steps;
        switch (result.status)
        {
        case solver_status::converged:
            result.message = "native L-BFGS converged";
            break;
        case solver_status::numerical_failure:
            result.message = out.message.empty() ? "native L-BFGS failed" : out.message;
            break;
        case solver_status::stalled:
            result.message = "native L-BFGS stalled: " + out.message;
            break;
        default:
            result.message = "native L-BFGS reached iteration limit";
            break;
        }
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("native L-BFGS threw: ") + e.what();
    }
    result.objective_evaluations = evaluator.counters().objective_evaluations;
    result.gradient_evaluations  = evaluator.counters().gradient_evaluations;
    return result;
}

// Ipopt general-objective path (bound-constrained NLP).
solver_result run_ipopt(const optimization_problem& problem,
    const vector_type&                              initial_guess,
    const solve_options&                            options)
{
    const ipopt_options cfg = options.ipopt.value_or(ipopt_options{});

    solver_result result;
    result.parameters = initial_guess;
    result.backend    = backend::ipopt;
    result.algorithm  = algorithm::interior_point;

    if (!ipopt_solver::is_supported())
    {
        result.status  = solver_status::backend_unavailable;
        result.message = "Ipopt backend was not compiled in (SOLVERS_ENABLE_IPOPT=OFF)";
        return result;
    }

    if (!problem.constraints.empty())
    {
        result.status  = solver_status::unsupported_capability;
        result.message = "Ipopt adapter does not support general nonlinear constraints; only box "
                         "bounds are implemented";
        return result;
    }

    const bool want_exact = cfg.hessian_mode == ipopt_hessian_mode::exact && problem.hessian;

    auto ipopt_opts = solver_options_ipopt_builder()
                          .with_max_iterations(options.max_iterations)
                          .with_tol(cfg.tol)
                          .with_acceptable_tol(cfg.acceptable_tol)
                          .with_max_wall_time_seconds(cfg.max_wall_time_seconds)
                          .with_linear_solver(cfg.linear_solver)
                          .with_hessian_approximation(
                              want_exact ? ipopt_hessian_approximation_enum::EXACT
                                         : ipopt_hessian_approximation_enum::LIMITED_MEMORY)
                          .with_verbose(options.verbose)
                          .build();

    auto resolution = detail::resolve_gradient(problem, options, initial_guess);
    if (auto* error = std::get_if<solver_result>(&resolution))
    {
        error->backend   = backend::ipopt;
        error->algorithm = algorithm::interior_point;
        return std::move(*error);
    }
    auto& resolved                     = std::get<detail::resolved_gradient>(resolution);
    result.effective_derivative_source = resolved.source;

    std::vector<double> parameters(
        initial_guess.data(), initial_guess.data() + initial_guess.size());

    ipopt_solver::hessian_type hess =
        problem.hessian ? *problem.hessian : ipopt_solver::hessian_type{};

    ipopt_solver solver(problem.num_parameters,
        resolved.objective,
        resolved.gradient,
        hess,
        problem.bounds.lower,
        problem.bounds.upper);

    backend_solve_status outcome;
    try
    {
        outcome = solver.solve_with_status(parameters, *ipopt_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("Ipopt threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    result.objective      = resolved.objective(result.parameters);
    result.status         = translate_outcome(outcome.outcome);
    result.backend_status = outcome.native_code;
    result.message        = outcome.message;
    if (outcome.iterations)
    {
        result.iterations = *outcome.iterations;
    }
    result.objective_evaluations = resolved.evaluator->counters().objective_evaluations;
    result.gradient_evaluations  = resolved.evaluator->counters().gradient_evaluations;
    return result;
}

// PETSc/TAO general-objective path.
solver_result run_petsc_tao_objective(const optimization_problem& problem,
    const vector_type&                                            initial_guess,
    const solve_options&                                          options)
{
    const petsc_tao_options cfg = options.petsc_tao.value_or(petsc_tao_options{});

    solver_result result;
    result.parameters = initial_guess;
    result.backend    = backend::petsc_tao;
    result.algorithm  = algorithm::newton_krylov;

    if (problem.hessian_vector)
    {
        result.status = solver_status::unsupported_capability;
        result.message =
            "PETSc/TAO matrix-free Hessian-vector execution is not implemented by this adapter";
        return result;
    }

    if (!petsc_tao_solver::is_supported())
    {
        result.status  = solver_status::backend_unavailable;
        result.message = "PETSc/TAO backend was not compiled in (SOLVERS_ENABLE_PETSC=OFF)";
        return result;
    }

    auto resolution = detail::resolve_gradient(problem, options, initial_guess);
    if (auto* error = std::get_if<solver_result>(&resolution))
    {
        error->backend   = backend::petsc_tao;
        error->algorithm = algorithm::newton_krylov;
        return std::move(*error);
    }
    auto& resolved                     = std::get<detail::resolved_gradient>(resolution);
    result.effective_derivative_source = resolved.source;

    auto petsc_opts = solver_options_petsc_builder()
                          .with_tao_type(map_tao_algorithm(cfg.algorithm, /*least_squares=*/false))
                          .with_max_iterations(options.max_iterations)
                          .with_gatol(cfg.gatol)
                          .with_grtol(cfg.grtol)
                          .with_matrix_free(cfg.matrix_free)
                          .with_verbose(options.verbose)
                          .build();

    std::vector<double> parameters(
        initial_guess.data(), initial_guess.data() + initial_guess.size());

    petsc_tao_solver::hessian_type hess =
        problem.hessian ? *problem.hessian : petsc_tao_solver::hessian_type{};

    petsc_tao_solver solver(problem.num_parameters,
        resolved.objective,
        resolved.gradient,
        hess,
        problem.bounds.lower,
        problem.bounds.upper);

    backend_solve_status outcome;
    try
    {
        outcome = solver.solve_with_status(parameters, *petsc_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("PETSc/TAO threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    result.objective      = resolved.objective(result.parameters);
    result.status         = translate_outcome(outcome.outcome);
    result.backend_status = outcome.native_code;
    result.message        = outcome.message;
    if (outcome.iterations)
    {
        result.iterations = *outcome.iterations;
    }
    result.objective_evaluations = resolved.evaluator->counters().objective_evaluations;
    result.gradient_evaluations  = resolved.evaluator->counters().gradient_evaluations;
    return result;
}
}  // namespace

solver_result solve(const least_squares_problem& problem,
    const vector_type&                           initial_guess,
    const solve_options&                         options)
{
    // -- validation (never evaluate a callback on an invalid request) --------
    if (problem.num_parameters == 0 || problem.num_residuals == 0 || !problem.residuals)
    {
        return failed(solver_status::invalid_problem,
            "least_squares_problem requires positive dimensions and a residual callback",
            initial_guess);
    }
    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }
    if (const auto message = detail::validate_bounds(problem.bounds, problem.num_parameters))
    {
        return failed(solver_status::invalid_problem, *message, initial_guess);
    }
    if (const auto error = detail::validate_options(options, initial_guess))
    {
        return failed(error->status, error->message, initial_guess);
    }

    const problem_traits traits   = inspect(problem);
    const auto           decision = detail::select_route(traits, options);
    if (decision.chosen == nullptr)
    {
        auto result      = failed(decision.failure, decision.message, initial_guess);
        result.backend   = options.backend;
        result.algorithm = options.algorithm;
        return result;
    }
    const api::backend   chosen = decision.chosen->backend;
    const api::algorithm alg    = decision.chosen->algorithm;

    switch (chosen)
    {
    case backend::native:
        return run_native_least_squares(problem, initial_guess, options, alg);
    case backend::ceres:
        return run_ceres(problem, initial_guess, options);
    case backend::pounders:
    case backend::petsc_tao:
        return run_petsc_tao_least_squares(problem, initial_guess, options, chosen, alg);
    default:
        break;
    }
    // Unreachable while the route table only lists the backends above.
    auto result      = failed(solver_status::unsupported_capability,
        std::string("backend '") + to_string(chosen) + "' does not serve the residual API",
        initial_guess);
    result.backend   = chosen;
    result.algorithm = alg;
    return result;
}

solver_result solve(const optimization_problem& problem,
    const vector_type&                          initial_guess,
    const solve_options&                        options)
{
    if (problem.num_parameters == 0 || !problem.objective)
    {
        return failed(solver_status::invalid_problem,
            "optimization_problem requires positive dimensions and an objective callback",
            initial_guess);
    }
    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }
    if (const auto message = detail::validate_bounds(problem.bounds, problem.num_parameters))
    {
        return failed(solver_status::invalid_problem, *message, initial_guess);
    }
    if (const auto error = detail::validate_options(options, initial_guess))
    {
        return failed(error->status, error->message, initial_guess);
    }

    const problem_traits traits   = inspect(problem);
    const auto           decision = detail::select_route(traits, options);
    if (decision.chosen == nullptr)
    {
        auto result      = failed(decision.failure, decision.message, initial_guess);
        result.backend   = options.backend;
        result.algorithm = options.algorithm;
        return result;
    }
    const api::backend   chosen = decision.chosen->backend;
    const api::algorithm alg    = decision.chosen->algorithm;

    switch (chosen)
    {
    case backend::ipopt:
        return run_ipopt(problem, initial_guess, options);
    case backend::petsc_tao:
        return run_petsc_tao_objective(problem, initial_guess, options);
    case backend::native:
        return run_native_optimization(problem, initial_guess, options, alg);
    default:
        break;
    }
    auto result      = failed(solver_status::unsupported_capability,
        std::string("backend '") + to_string(chosen) + "' does not serve general objectives",
        initial_guess);
    result.backend   = chosen;
    result.algorithm = alg;
    return result;
}
}  // namespace solverslib::api
