#include "solvers/api/solve.h"

#include <exception>
#include <string>
#include <vector>

#include "detail/native_result.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solvers/ceres_solver.h"

#if SOLVERS_HAS_CERES
#include <ceres/ceres.h>
#endif
#include "solvers/gauss_newton_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/lbfgs_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/petsc_tao_solver.h"

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
// Map a resolved algorithm to the backend that realizes it, when the caller
// left the backend on automatic.
backend backend_for(algorithm value, const problem_traits& traits)
{
    switch (value)
    {
    case algorithm::levenberg_marquardt:
    case algorithm::gauss_newton:
    case algorithm::bfgs:
    case algorithm::lbfgs:
        // Route to Ceres only when the provider is Ceres-native AD and no
        // other Jacobian (callable or generic provider) is available.
        if (traits.has_autodiff_provider && !traits.has_callable_jacobian &&
            !traits.has_jacobian_provider)
        {
            return backend::ceres;
        }
        return backend::native;
    case algorithm::pounders:
        return backend::pounders;
    case algorithm::interior_point:
        return backend::ipopt;
    case algorithm::newton_krylov:
        return backend::petsc_tao;
    case algorithm::automatic:
        return backend::automatic;
    }
    return backend::automatic;
}

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
    api::derivative_mode                       deriv_mode = api::derivative_mode::automatic)
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
    result.message = (result.status == solver_status::converged) ? "native converged"
                                                                 : "native reached iteration limit";
    return result;
}
}  // namespace

problem_traits inspect(const least_squares_problem& problem)
{
    problem_traits traits;
    traits.is_least_squares      = true;
    traits.has_callable_jacobian = problem.has_callable_jacobian();
    traits.has_jacobian_provider = !!problem.jacobian_provider;
    traits.has_autodiff_provider = !!problem.provider_factory;
    // has_jacobian is true when any derivative source is available
    traits.has_jacobian = traits.has_callable_jacobian || traits.has_jacobian_provider ||
                          problem.jacobian.has_value();
    traits.has_bounds     = !problem.bounds.empty();
    traits.num_parameters = problem.num_parameters;
    traits.num_residuals  = problem.num_residuals;
    return traits;
}

problem_traits inspect(const optimization_problem& problem)
{
    problem_traits traits;
    traits.is_least_squares           = false;
    traits.has_gradient_provider      = !!problem.gradient_provider;
    traits.has_gradient               = problem.gradient.has_value() || !!problem.gradient_provider;
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
    if (options.algorithm != algorithm::automatic)
    {
        return options.algorithm;
    }

    if (traits.is_least_squares)
    {
        if (!traits.has_jacobian)
        {
            return algorithm::pounders;  // derivative-free least squares
        }
        if (is_large_scale(traits, options.policy))
        {
            return algorithm::newton_krylov;  // matrix-free / large scale -> TAO
        }
        return algorithm::levenberg_marquardt;
    }

    // General objective.
    if (traits.has_nonlinear_constraints)
    {
        return algorithm::interior_point;  // IPOPT
    }
    if (is_large_scale(traits, options.policy))
    {
        return algorithm::newton_krylov;  // large scale -> TAO
    }
    return algorithm::lbfgs;
}

backend select_backend(const problem_traits& traits, const solve_options& options)
{
    if (options.backend != backend::automatic)
    {
        return options.backend;
    }
    return backend_for(select_algorithm(traits, options), traits);
}

namespace
{
// Result of derivative policy resolution
struct derivative_resolution
{
    derivative_mode resolved;
    solver_result*  error;  // nullptr if OK; otherwise result explaining incompatibility
};

// Resolve derivative policy to a concrete implementation.
// Uses provider source() instead of conflating all providers as "supplied".
// Applied to all backends for consistency.
derivative_resolution resolve_derivatives(
    const least_squares_problem& problem, const solve_options& options, const vector_type& x)
{
    auto policy = options.derivatives;

    // Rule 1: explicit request takes precedence, with strict validation
    if (policy != derivative_mode::automatic)
    {
        if (policy == derivative_mode::supplied)
        {
            // Accept either an explicit jacobian callback or a generic provider
            // Crucially: a provider is only "supplied" if its source() says so
            const bool has_supplied_source =
                problem.has_callable_jacobian() ||
                (problem.jacobian_provider &&
                    problem.jacobian_provider->source() == derivative_mode::supplied);

            if (!has_supplied_source)
            {
                auto err = new solver_result();
                *err     = failed(solver_status::invalid_problem,
                    "supplied Jacobian required but not provided",
                    x);
                return {policy, err};
            }
        }
        else if (policy == derivative_mode::automatic_differentiation)
        {
            // Require an executable AD provider
            if (!problem.provider_factory &&
                (!problem.jacobian_provider || problem.jacobian_provider->source() !=
                                                   derivative_mode::automatic_differentiation))
            {
                auto err = new solver_result();
                *err     = failed(solver_status::unsupported_capability,
                    "automatic differentiation required but no compatible AD provider available",
                    x);
                return {policy, err};
            }
        }
        else if (policy == derivative_mode::finite_difference)
        {
            // Explicit finite differences: use them even if other sources exist
            // Validation happens at solver boundary (e.g., reject if no residuals callable)
        }
        return {policy, nullptr};
    }

    // Rule 2: automatic cascade with truthful source reporting
    // Supplied (explicit Jacobian) > Provider with its actual source > Fallback
    if (problem.has_callable_jacobian())
    {
        return {derivative_mode::supplied, nullptr};
    }

    if (problem.jacobian_provider)
    {
        // Use the provider's actual reported source, not "supplied"
        return {problem.jacobian_provider->source(), nullptr};
    }

    if (problem.provider_factory)
    {
        // Ceres-native AD factory present; will use Ceres if that backend is selected
        return {derivative_mode::automatic_differentiation, nullptr};
    }

    // Otherwise fall through to backend's default (Ceres numeric, native finite-diff, etc.)
    return {derivative_mode::automatic, nullptr};
}

// Shared native least-squares execution once validation has passed.
solver_result run_native_least_squares(const least_squares_problem& problem,
    const vector_type&                                              initial_guess,
    const solve_options&                                            options,
    algorithm                                                       alg)
{
    vector_type x = initial_guess;

    // Resolve derivative policy for native backend
    auto res = resolve_derivatives(problem, options, initial_guess);
    if (res.error)
    {
        solver_result error_result = *res.error;
        delete res.error;
        return error_result;
    }
    auto resolved_derivatives = res.resolved;

    // Build the Jacobian callback for native kernels.
    // Use the resolved derivative source to guide construction.
    jacobian_function jac;

    if (resolved_derivatives != derivative_mode::finite_difference && problem.jacobian_provider)
    {
        // Wrap the provider so native solvers see a standard jacobian_function.
        auto provider_ptr = problem.jacobian_provider;
        jac               = [provider_ptr](const vector_type& x_, matrix_type& J_)
        {
            vector_type r_tmp = make_vector(provider_ptr->num_residuals());
            provider_ptr->compute(x_, r_tmp, J_);
        };
    }
    else
    {
        // Explicit callback, or null so the native solver uses finite differences.
        jac = problem.jacobian.value_or(jacobian_function{});
    }

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
        gauss_newton_solver solver(
            problem.num_parameters, problem.num_residuals, problem.residuals, jac);
        return from_native(solver.solve(x, *native_opts), x, alg, resolved_derivatives);
    }
    case algorithm::bfgs:
    case algorithm::lbfgs:
    {
        auto native_opts = solver_options_bfgs_builder()
                               .with_max_iterations(options.max_iterations)
                               .with_function_tolerance(options.function_tolerance)
                               .with_gradient_tolerance(options.gradient_tolerance)
                               .with_parameter_tolerance(options.parameter_tolerance)
                               .with_verbose(options.verbose)
                               .build();
        lbfgs_solver solver(problem.num_parameters, problem.num_residuals, problem.residuals, jac);
        return from_native(
            solver.solve(x, *native_opts), x, algorithm::lbfgs, resolved_derivatives);
    }
    case algorithm::levenberg_marquardt:
    default:
    {
        auto native_opts = solver_options_lm_builder()
                               .with_max_iterations(options.max_iterations)
                               .with_function_tolerance(options.function_tolerance)
                               .with_gradient_tolerance(options.gradient_tolerance)
                               .with_parameter_tolerance(options.parameter_tolerance)
                               .with_verbose(options.verbose)
                               .build();
        levenberg_marquardt_solver solver(
            problem.num_parameters, problem.num_residuals, problem.residuals, jac);
        return from_native(
            solver.solve(x, *native_opts), x, algorithm::levenberg_marquardt, resolved_derivatives);
    }
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

    // Resolve derivative policy
    auto res = resolve_derivatives(problem, options, initial_guess);
    if (res.error)
    {
        // Error occurred during resolution
        solver_result error_result = *res.error;
        delete res.error;
        return error_result;
    }
    auto resolved_derivatives = res.resolved;

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

    // Select the derivative source for the Ceres backend.
    std::shared_ptr<const api::detail::provider_factory>  provider_to_use;
    std::function<void(const vector_type&, matrix_type&)> jacobian_to_use;

    if (resolved_derivatives == api::derivative_mode::automatic_differentiation)
    {
        if (problem.provider_factory)
        {
            // Ceres-native AD: the factory produces a CostFunction using Jet.
            provider_to_use = problem.provider_factory;
        }
        else if (problem.jacobian_provider && problem.jacobian_provider->source() ==
                                                  api::derivative_mode::automatic_differentiation)
        {
            // AD provider without a Ceres factory: wrap compute() as a Jacobian callback.
            auto jp         = problem.jacobian_provider;
            jacobian_to_use = [jp](const vector_type& x_, matrix_type& J_)
            {
                vector_type r_tmp = make_vector(jp->num_residuals());
                jp->compute(x_, r_tmp, J_);
            };
        }
    }
    else if (resolved_derivatives == api::derivative_mode::supplied)
    {
        // Only use a provider whose source() is actually "supplied"; otherwise
        // fall through to the explicit Jacobian callback so the resolved mode is honoured.
        if (problem.jacobian_provider &&
            problem.jacobian_provider->source() == api::derivative_mode::supplied)
        {
            auto jp         = problem.jacobian_provider;
            jacobian_to_use = [jp](const vector_type& x_, matrix_type& J_)
            {
                vector_type r_tmp = make_vector(jp->num_residuals());
                jp->compute(x_, r_tmp, J_);
            };
        }
        else if (problem.jacobian)
        {
            jacobian_to_use = *problem.jacobian;
        }
    }
    // Otherwise: no explicit derivative source; Ceres falls back to finite differences.

    ceres_solver solver(problem.num_parameters,
        problem.num_residuals,
        problem.residuals,
        provider_to_use,
        problem.bounds.lower,
        problem.bounds.upper);

    if (jacobian_to_use && !provider_to_use)
    {
        solver = ceres_solver(problem.num_parameters,
            problem.num_residuals,
            problem.residuals,
            jacobian_to_use,
            problem.bounds.lower,
            problem.bounds.upper);
    }

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

    // Map Ceres termination to solver status truthfully
    if (summary.termination_type == ceres::TerminationType::CONVERGENCE)
    {
        result.status  = solver_status::converged;
        result.message = "Ceres converged";
    }
    else if (summary.termination_type == ceres::TerminationType::NO_CONVERGENCE)
    {
        // Distinguish: hit iteration limit vs. failure
        if (summary.IsSolutionUsable() && summary.iterations.size() > 0)
        {
            result.status  = solver_status::max_iterations;
            result.message = "Ceres reached maximum iterations (solution usable)";
        }
        else
        {
            result.status  = solver_status::numerical_failure;
            result.message = "Ceres did not converge";
        }
    }
    else
    {
        result.status  = solver_status::numerical_failure;
        result.message = "Ceres failed";
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
    backend                                                            chosen)
{
    const petsc_tao_options cfg = options.petsc_tao.value_or(petsc_tao_options{});

    solver_result result;
    result.parameters = initial_guess;
    result.backend    = chosen;
    result.algorithm = chosen == backend::pounders ? algorithm::pounders : algorithm::newton_krylov;

    if (!petsc_tao_solver::is_supported())
    {
        result.status  = solver_status::backend_unavailable;
        result.message = "PETSc/TAO backend was not compiled in (SOLVERS_ENABLE_PETSC=OFF)";
        return result;
    }

    // Resolve derivative policy for TAO backend
    auto res = resolve_derivatives(problem, options, initial_guess);
    if (res.error)
    {
        solver_result error_result = *res.error;
        delete res.error;
        return error_result;
    }
    auto resolved_derivatives = res.resolved;

    // POUNDERS by default on the pounders route; BRGN default on the general
    // TAO route (Gauss-Newton for large residuals with a Jacobian).
    tao_algorithm requested = cfg.algorithm;
    if (requested == tao_algorithm::automatic)
    {
        requested = chosen == backend::pounders ? tao_algorithm::pounders : tao_algorithm::brgn;
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

    // Build Jacobian callback, preferring provider over callback
    petsc_tao_solver::jacobian_type jac;
    if (problem.jacobian_provider)
    {
        auto provider_ptr = problem.jacobian_provider;
        jac               = [provider_ptr](const vector_type& x_, matrix_type& J_)
        {
            vector_type r_tmp = make_vector(provider_ptr->num_residuals());
            provider_ptr->compute(x_, r_tmp, J_);
        };
    }
    else
    {
        jac = problem.jacobian ? *problem.jacobian : petsc_tao_solver::jacobian_type{};
    }

    petsc_tao_solver solver(problem.num_parameters,
        problem.num_residuals,
        problem.residuals,
        jac,
        problem.bounds.lower,
        problem.bounds.upper);

    bool converged = false;
    try
    {
        converged = solver.solve(parameters, *petsc_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("PETSc/TAO threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    const double rnorm   = residual_norm_at(problem, result.parameters);
    result.residual_norm = rnorm;
    result.objective     = 0.5 * rnorm * rnorm;
    result.status        = converged ? solver_status::converged : solver_status::max_iterations;
    result.effective_derivative_source = resolved_derivatives;
    result.message = converged ? "PETSc/TAO converged" : "PETSc/TAO stopped without convergence";
    return result;
}

// Native scalar-objective path via L-BFGS.
solver_result run_native_optimization(const optimization_problem& problem,
    const vector_type&                                            initial_guess,
    const solve_options&                                          options,
    algorithm                                                     alg)
{
    // Build gradient callback: gradient_provider > gradient callback > error.
    gradient_function grad;
    if (problem.gradient_provider)
    {
        auto gp = problem.gradient_provider;
        grad    = [gp](const vector_type& x_, vector_type& g_) { gp->compute(x_, g_); };
    }
    else if (problem.gradient)
    {
        grad = *problem.gradient;
    }
    else
    {
        solver_result result = failed(solver_status::unsupported_capability,
            "native L-BFGS requires a gradient callback or gradient provider",
            initial_guess);
        result.backend       = backend::native;
        result.algorithm     = alg;
        return result;
    }

    vector_type x = initial_guess;

    auto native_opts = solver_options_bfgs_builder()
                           .with_max_iterations(options.max_iterations)
                           .with_function_tolerance(options.function_tolerance)
                           .with_gradient_tolerance(options.gradient_tolerance)
                           .with_parameter_tolerance(options.parameter_tolerance)
                           .with_verbose(options.verbose)
                           .build();

    lbfgs_solver solver(problem.num_parameters, problem.objective, grad);
    auto         out = solver.solve(x, *native_opts);

    solver_result result;
    result.status     = translate_native(out.status);
    result.parameters = x;
    result.objective  = problem.objective(x);
    result.iterations = out.iterations;
    result.backend    = backend::native;
    result.algorithm  = algorithm::lbfgs;
    result.message    = (result.status == solver_status::converged)
                            ? "native L-BFGS converged"
                            : "native L-BFGS reached iteration limit";
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

    // Build gradient for Ipopt: gradient_provider > gradient callback > error.
    gradient_function grad_for_ipopt;
    if (problem.gradient_provider)
    {
        auto gp        = problem.gradient_provider;
        grad_for_ipopt = [gp](const vector_type& x_, vector_type& g_) { gp->compute(x_, g_); };
    }
    else if (problem.gradient)
    {
        grad_for_ipopt = *problem.gradient;
    }
    else
    {
        result.status  = solver_status::unsupported_capability;
        result.message = "Ipopt path requires a gradient callback or gradient provider";
        return result;
    }

    std::vector<double> parameters(
        initial_guess.data(), initial_guess.data() + initial_guess.size());

    ipopt_solver::hessian_type hess =
        problem.hessian ? *problem.hessian : ipopt_solver::hessian_type{};

    ipopt_solver solver(problem.num_parameters,
        problem.objective,
        grad_for_ipopt,
        hess,
        problem.bounds.lower,
        problem.bounds.upper);

    bool ok = false;
    try
    {
        ok = solver.solve(parameters, *ipopt_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("Ipopt threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    result.objective = problem.objective(result.parameters);
    result.status    = ok ? solver_status::converged : solver_status::numerical_failure;
    result.message   = ok ? "Ipopt reached an (acceptable) solution" : "Ipopt did not converge";
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

    if (!petsc_tao_solver::is_supported())
    {
        result.status  = solver_status::backend_unavailable;
        result.message = "PETSc/TAO backend was not compiled in (SOLVERS_ENABLE_PETSC=OFF)";
        return result;
    }

    // Build gradient for TAO: gradient_provider > gradient callback > error.
    gradient_function grad_for_tao;
    if (problem.gradient_provider)
    {
        auto gp      = problem.gradient_provider;
        grad_for_tao = [gp](const vector_type& x_, vector_type& g_) { gp->compute(x_, g_); };
    }
    else if (problem.gradient)
    {
        grad_for_tao = *problem.gradient;
    }
    else
    {
        result.status  = solver_status::unsupported_capability;
        result.message = "PETSc/TAO path requires a gradient callback or gradient provider";
        return result;
    }

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
        problem.objective,
        grad_for_tao,
        hess,
        problem.bounds.lower,
        problem.bounds.upper);

    bool converged = false;
    try
    {
        converged = solver.solve(parameters, *petsc_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("PETSc/TAO threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    result.objective = problem.objective(result.parameters);
    result.status    = converged ? solver_status::converged : solver_status::max_iterations;
    result.message   = converged ? "PETSc/TAO converged" : "PETSc/TAO stopped without convergence";
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

    const problem_traits traits = inspect(problem);
    const api::backend   chosen = select_backend(traits, options);
    const api::algorithm alg    = select_algorithm(traits, options);

    switch (chosen)
    {
    case backend::native:
    {
        // Native kernels do not enforce bounds; reject rather than solve the
        // unconstrained relaxation (review F01).
        if (traits.has_bounds)
        {
            solver_result result = failed(solver_status::unsupported_capability,
                "native backend does not enforce bounds; use a bound-capable backend",
                initial_guess);
            result.backend       = backend::native;
            result.algorithm     = alg;
            return result;
        }
        return run_native_least_squares(problem, initial_guess, options, alg);
    }
    case backend::ceres:
        return run_ceres(problem, initial_guess, options);
    case backend::pounders:
    case backend::petsc_tao:
        return run_petsc_tao_least_squares(problem, initial_guess, options, chosen);
    default:
    {
        // Ipopt is a general-NLP solver; it does not serve the least-squares
        // residual API directly.
        solver_result result = failed(solver_status::unsupported_capability,
            std::string("backend '") + to_string(chosen) +
                "' does not serve the least-squares residual API",
            initial_guess);
        result.backend       = chosen;
        result.algorithm     = alg;
        return result;
    }
    }
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

    const problem_traits traits = inspect(problem);
    const api::backend   chosen = select_backend(traits, options);
    const api::algorithm alg    = select_algorithm(traits, options);

    switch (chosen)
    {
    case backend::ipopt:
        return run_ipopt(problem, initial_guess, options);
    case backend::petsc_tao:
        return run_petsc_tao_objective(problem, initial_guess, options);
    case backend::native:
    {
        if (traits.has_bounds)
        {
            solver_result result = failed(solver_status::unsupported_capability,
                "native backend does not enforce bounds; use a bound-capable backend",
                initial_guess);
            result.backend       = backend::native;
            result.algorithm     = alg;
            return result;
        }
        return run_native_optimization(problem, initial_guess, options, alg);
    }
    default:
    {
        solver_result result = failed(solver_status::unsupported_capability,
            std::string("backend '") + to_string(chosen) +
                "' does not serve general objective optimization",
            initial_guess);
        result.backend       = chosen;
        result.algorithm     = alg;
        return result;
    }
    }
}
}  // namespace solverslib::api
