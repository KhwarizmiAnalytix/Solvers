#include "solvers/api/solve.h"

#include <cmath>
#include <exception>
#include <string>
#include <vector>

#include "detail/backend_status.h"
#include "detail/native_result.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solvers/gauss_newton_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/lbfgs_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/petsc_tao_solver.h"
#include "solvers/rnc_lm_solver.h"

namespace solverslib::api
{
namespace
{
// Status translation helpers
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

// Helper to create a failed result
solver_result failed(solver_status status, const std::string& message, const vector_type& x)
{
    solver_result result;
    result.status     = status;
    result.parameters = x;
    result.message    = message;
    return result;
}

// Residual L2 norm at a point
double residual_norm_at(const least_squares_problem& problem, const vector_type& x)
{
    vector_type r = make_vector(problem.num_residuals);
    problem.residuals(x, r);
    return r.norm();
}
}  // namespace

// Inspect least-squares problem structure
problem_traits inspect(const least_squares_problem& problem)
{
    problem_traits traits;
    traits.is_least_squares = true;
    traits.has_jacobian     = problem.jacobian.has_value();
    traits.has_bounds       = !problem.bounds.empty();
    traits.num_parameters   = problem.num_parameters;
    traits.num_residuals    = problem.num_residuals;
    return traits;
}

// Inspect optimization problem structure
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

namespace
{
// Native least-squares solver
solver_result solve_native_least_squares(
    const least_squares_problem& problem,
    const vector_type&           initial_guess,
    const solve_options&         options,
    algorithm                    alg)
{
    vector_type x = initial_guess;
    solver_result result;
    result.parameters = x;
    result.backend    = backend::native;
    result.algorithm  = alg;

    try
    {
        switch (alg)
        {
        case algorithm::levenberg_marquardt:
        case algorithm::automatic:
        {
            auto native_opts = solver_options_lm_builder()
                                   .with_max_iterations(options.max_iterations)
                                   .with_function_tolerance(options.function_tolerance)
                                   .with_gradient_tolerance(options.gradient_tolerance)
                                   .with_parameter_tolerance(options.parameter_tolerance)
                                   .with_verbose(options.verbose)
                                   .build();

            levenberg_marquardt_solver solver(
                problem.num_parameters, problem.num_residuals, problem.residuals, nullptr);
            const auto out = solver.solve(x, *native_opts);

            result.status        = translate_native(out.status);
            result.parameters    = x;
            result.residual_norm = out.residual_norm;
            result.objective     = 0.5 * out.residual_norm * out.residual_norm;
            result.iterations    = out.iterations;
            result.gradient_norm = out.gradient_norm;
            result.step_norm     = out.step_norm;
            result.accepted_steps = out.accepted_steps;
            result.rejected_steps = out.rejected_steps;
            result.message       = out.message.empty() ? "native converged" : out.message;
            return result;
        }
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
                problem.num_parameters, problem.num_residuals, problem.residuals, nullptr);
            const auto out = solver.solve(x, *native_opts);

            result.status        = translate_native(out.status);
            result.parameters    = x;
            result.residual_norm = out.residual_norm;
            result.objective     = 0.5 * out.residual_norm * out.residual_norm;
            result.iterations    = out.iterations;
            result.gradient_norm = out.gradient_norm;
            result.step_norm     = out.step_norm;
            result.accepted_steps = out.accepted_steps;
            result.rejected_steps = out.rejected_steps;
            result.message       = out.message.empty() ? "native converged" : out.message;
            return result;
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

            lbfgs_solver solver(
                problem.num_parameters, problem.num_residuals, problem.residuals, nullptr);
            const auto out = solver.solve(x, *native_opts);

            result.status        = translate_native(out.status);
            result.parameters    = x;
            result.residual_norm = out.residual_norm;
            result.objective     = 0.5 * out.residual_norm * out.residual_norm;
            result.iterations    = out.iterations;
            result.gradient_norm = out.gradient_norm;
            result.step_norm     = out.step_norm;
            result.accepted_steps = out.accepted_steps;
            result.rejected_steps = out.rejected_steps;
            result.message       = out.message.empty() ? "native converged" : out.message;
            return result;
        }
        case algorithm::riemann_normal_coordinate_lm:
        {
            return solve_rnc_lm(problem, initial_guess, options);
        }
        default:
            result.status  = solver_status::unsupported_capability;
            result.message = "algorithm not supported";
            return result;
        }
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("native solver threw: ") + e.what();
        return result;
    }
}

// Native optimization solver
solver_result solve_native_optimization(
    const optimization_problem& problem,
    const vector_type&          initial_guess,
    const solve_options&        options)
{
    vector_type x = initial_guess;

    auto native_opts = solver_options_bfgs_builder()
                           .with_max_iterations(options.max_iterations)
                           .with_function_tolerance(options.function_tolerance)
                           .with_gradient_tolerance(options.gradient_tolerance)
                           .with_parameter_tolerance(options.parameter_tolerance)
                           .with_verbose(options.verbose)
                           .build();

    solver_result result;
    result.parameters = x;
    result.backend    = backend::native;
    result.algorithm  = algorithm::lbfgs;

    try
    {
        lbfgs_solver solver(problem.num_parameters, problem.objective, nullptr);
        const auto   out = solver.solve(x, *native_opts);

        result.status        = translate_native(out.status);
        result.iterations    = out.iterations;
        result.parameters    = x;
        result.objective     = problem.objective(x);
        result.gradient_norm = out.gradient_norm;
        result.step_norm     = out.step_norm;
        result.accepted_steps = out.accepted_steps;
        result.rejected_steps = out.rejected_steps;
        result.message       = out.message.empty() ? "native converged" : out.message;
        return result;
    }
    catch (const std::exception& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = std::string("native L-BFGS threw: ") + e.what();
        return result;
    }
}
}  // namespace

// Least-squares solve entry point
solver_result solve(
    const least_squares_problem& problem,
    const vector_type&           initial_guess,
    const solve_options&         options)
{
    // Validation
    if (problem.num_parameters == 0 || problem.num_residuals == 0 || !problem.residuals)
    {
        return failed(
            solver_status::invalid_problem,
            "least_squares_problem requires positive dimensions and a residual callback",
            initial_guess);
    }
    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(
            solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }
    if (!problem.bounds.empty() && problem.bounds.lower.size() != problem.num_parameters &&
        problem.bounds.upper.size() != problem.num_parameters)
    {
        return failed(
            solver_status::invalid_problem,
            "bounds dimensions do not match num_parameters",
            initial_guess);
    }

    // Determine algorithm: automatic defaults to LM
    auto chosen_algorithm = options.algorithm;
    if (chosen_algorithm == algorithm::automatic)
    {
        chosen_algorithm = algorithm::levenberg_marquardt;
    }

    // Route to native solver
    if (options.backend == backend::automatic || options.backend == backend::native)
    {
        return solve_native_least_squares(problem, initial_guess, options, chosen_algorithm);
    }

    // Other backends not yet implemented
    return failed(
        solver_status::backend_unavailable,
        "only native backend is currently implemented",
        initial_guess);
}

// Optimization solve entry point
solver_result solve(
    const optimization_problem& problem,
    const vector_type&          initial_guess,
    const solve_options&        options)
{
    // Validation
    if (problem.num_parameters == 0 || !problem.objective)
    {
        return failed(
            solver_status::invalid_problem,
            "optimization_problem requires positive dimensions and an objective callback",
            initial_guess);
    }
    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(
            solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }
    if (!problem.bounds.empty() && problem.bounds.lower.size() != problem.num_parameters &&
        problem.bounds.upper.size() != problem.num_parameters)
    {
        return failed(
            solver_status::invalid_problem,
            "bounds dimensions do not match num_parameters",
            initial_guess);
    }

    // Route to native solver
    if (options.backend == backend::automatic || options.backend == backend::native)
    {
        return solve_native_optimization(problem, initial_guess, options);
    }

    // Other backends not yet implemented
    return failed(
        solver_status::backend_unavailable,
        "only native backend is currently implemented",
        initial_guess);
}
}  // namespace solverslib::api
