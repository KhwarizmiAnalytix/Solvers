#include "solvers/api/solve.h"

#include <exception>
#include <string>
#include <vector>

#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "detail/native_result.h"
#include "solvers/ceres_solver.h"
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
backend backend_for(algorithm value)
{
    switch (value)
    {
    case algorithm::levenberg_marquardt:
    case algorithm::gauss_newton:
    case algorithm::bfgs:
    case algorithm::lbfgs:
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

solver_result failed(solver_status status, std::string message, const vector_type& x)
{
    solver_result result;
    result.status     = status;
    result.parameters = x;
    result.message    = std::move(message);
    return result;
}

solver_result from_native(const native_result& out, const vector_type& x, api::algorithm alg)
{
    solver_result result;
    result.status         = translate_native(out.status);
    result.parameters     = x;
    result.residual_norm  = out.residual_norm;
    result.objective      = 0.5 * out.residual_norm * out.residual_norm;
    result.iterations     = out.iterations;
    result.backend        = backend::native;
    result.algorithm      = alg;
    result.backend_status = static_cast<int>(out.status);
    result.message        = (result.status == solver_status::converged) ? "native converged"
                                                                        : "native reached iteration limit";
    return result;
}
}  // namespace

problem_traits inspect(const least_squares_problem& problem)
{
    problem_traits traits;
    traits.is_least_squares      = true;
    traits.has_jacobian          = problem.jacobian.has_value();
    traits.has_callable_jacobian = problem.has_callable_jacobian();
    traits.has_autodiff_provider = !!problem.provider_factory;
    traits.has_bounds            = !problem.bounds.empty();
    traits.num_parameters        = problem.num_parameters;
    traits.num_residuals         = problem.num_residuals;
    return traits;
}

problem_traits inspect(const optimization_problem& problem)
{
    problem_traits traits;
    traits.is_least_squares           = false;
    traits.has_gradient               = problem.gradient.has_value();
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
    return backend_for(select_algorithm(traits, options));
}

namespace
{
// Result of derivative policy resolution
struct derivative_resolution
{
    derivative_mode resolved;
    solver_result*  error;  // nullptr if OK; otherwise result explaining incompatibility
};

// Resolve derivative policy to a concrete implementation
derivative_resolution resolve_derivatives(
    const least_squares_problem& problem,
    const solve_options& options,
    backend chosen_backend,
    const vector_type& x)
{
    auto policy = options.derivatives;

    // Rule 1: explicit request takes precedence
    if (policy != derivative_mode::automatic)
    {
        // Validate that the backend can execute the requested policy
        if (policy == derivative_mode::supplied && !problem.has_callable_jacobian())
        {
            auto err = new solver_result();
            *err = failed(solver_status::invalid_problem,
                "supplied Jacobian required but not provided", x);
            return {policy, err};
        }
        if (policy == derivative_mode::automatic_differentiation)
        {
            // Check that a provider exists and backend is compatible
            if (!problem.provider_factory)
            {
                auto err = new solver_result();
                *err = failed(solver_status::unsupported_capability,
                    "automatic differentiation required but no AD provider available", x);
                return {policy, err};
            }
            if (chosen_backend != backend::ceres)
            {
                auto err = new solver_result();
                *err = failed(solver_status::unsupported_capability,
                    "AD required but selected backend is not Ceres", x);
                return {policy, err};
            }
        }
        return {policy, nullptr};
    }

    // Rule 2: automatic mode cascades
    // Supplied > AD > numeric/derivative-free
    if (problem.has_callable_jacobian())
    {
        return {derivative_mode::supplied, nullptr};
    }
    if (problem.provider_factory && chosen_backend == backend::ceres)
    {
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

    // Native kernels take a null jacobian to mean "estimate by finite
    // differences", matching problem.jacobian == std::nullopt.
    jacobian_function jac = problem.jacobian.value_or(jacobian_function{});

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
        return from_native(solver.solve(x, *native_opts), x, alg);
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
        return from_native(solver.solve(x, *native_opts), x, algorithm::lbfgs);
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
        return from_native(solver.solve(x, *native_opts), x, algorithm::levenberg_marquardt);
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
    auto res = resolve_derivatives(problem, options, backend::ceres, initial_guess);
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

    // Create solver using appropriate path based on derivative resolution
    ceres_solver solver(problem.num_parameters,
        problem.num_residuals,
        problem.residuals,
        problem.provider_factory,
        problem.bounds.lower,
        problem.bounds.upper);

    // If no provider but we have a Jacobian callback, set it
    if (!problem.provider_factory && problem.jacobian)
    {
        // Use legacy constructor signature
        ceres_solver::CostFunctionLambda_aad jac = *problem.jacobian;
        solver = ceres_solver(problem.num_parameters,
            problem.num_residuals,
            problem.residuals,
            jac,
            problem.bounds.lower,
            problem.bounds.upper);
    }

    bool usable = false;
    try
    {
        usable = solver.solve(parameters, *ceres_opts);
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("Ceres threw: ") + e.what();
        return result;
    }

    result.parameters =
        to_vector_type(parameters.data(), static_cast<std::size_t>(parameters.size()));
    const double rnorm   = residual_norm_at(problem, result.parameters);
    result.residual_norm = rnorm;
    result.objective     = 0.5 * rnorm * rnorm;
    result.status        = usable ? solver_status::converged : solver_status::numerical_failure;
    result.message = usable ? "Ceres returned a usable solution" : "Ceres solution not usable";
    result.effective_derivative_source = resolved_derivatives;
    return result;
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

    petsc_tao_solver::jacobian_type jac =
        problem.jacobian ? *problem.jacobian : petsc_tao_solver::jacobian_type{};

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
    result.message = converged ? "PETSc/TAO converged" : "PETSc/TAO stopped without convergence";
    return result;
}

// Native scalar-objective path via L-BFGS.
solver_result run_native_optimization(const optimization_problem& problem,
    const vector_type&                                            initial_guess,
    const solve_options&                                          options,
    algorithm                                                     alg)
{
    if (!problem.gradient)
    {
        solver_result result = failed(solver_status::unsupported_capability,
            "native L-BFGS requires a gradient callback on the optimization_problem",
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

    lbfgs_solver solver(problem.num_parameters, problem.objective, *problem.gradient);
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

    // Ipopt needs a gradient; fall back to nothing only if the caller omitted
    // it (a future evaluator service would supply finite differences here).
    if (!problem.gradient)
    {
        result.status  = solver_status::unsupported_capability;
        result.message = "Ipopt path requires a gradient callback on the optimization_problem";
        return result;
    }

    std::vector<double> parameters(
        initial_guess.data(), initial_guess.data() + initial_guess.size());

    ipopt_solver::hessian_type hess =
        problem.hessian ? *problem.hessian : ipopt_solver::hessian_type{};

    ipopt_solver solver(problem.num_parameters,
        problem.objective,
        *problem.gradient,
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
    if (!problem.gradient)
    {
        result.status  = solver_status::unsupported_capability;
        result.message = "PETSc/TAO path requires a gradient callback on the optimization_problem";
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
        *problem.gradient,
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
