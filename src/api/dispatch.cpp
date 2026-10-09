#include "api/dispatch.h"

#include <functional>
#include <string>
#include <vector>

#include "detail/backend_status.h"
#include "solver_options/solver_options.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/ceres_solver.h"
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
using jacobian_type = std::function<void(const vector_type&, matrix_type&)>;
using gradient_type = std::function<void(const vector_type&, vector_type&)>;

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

solver_result failed(solver_status status, const std::string& message, const vector_type& x)
{
    solver_result result;
    result.status     = status;
    result.parameters = x;
    result.message    = message;
    return result;
}

vector_type to_vector(const std::vector<double>& values)
{
    return Eigen::Map<const vector_type>(values.data(), static_cast<index_type>(values.size()));
}

std::vector<double> to_std(const vector_type& values)
{
    return std::vector<double>(values.data(), values.data() + values.size());
}

solver_status translate_backend(backend_outcome outcome)
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

solver_result from_backend(const backend_solve_status& out, const std::vector<double>& x, backend which)
{
    solver_result result;
    result.parameters     = to_vector(x);
    result.status         = translate_backend(out.outcome);
    result.message        = out.message;
    result.backend        = which;
    result.backend_status = out.native_code;
    if (out.iterations)
    {
        result.iterations = *out.iterations;
    }
    return result;
}

algorithm resolve_algorithm(const solver_options& options)
{
    // Keyed on the options type: the native LS branch static_casts `options`
    // to the matching concrete class, so the algorithm must agree with it.
    switch (options.solver())
    {
    case solver_enum::LM:
        return dynamic_cast<const solver_options_rnc_lm*>(&options)
                   ? algorithm::riemann_normal_coordinate_lm
                   : algorithm::levenberg_marquardt;
    case solver_enum::GAUSS_NEWTON:
        return algorithm::gauss_newton;
    case solver_enum::LBFGS:
        return algorithm::lbfgs;
    default:
        return algorithm::automatic;
    }
}
}

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

problem_traits inspect(const optimization_problem& problem)
{
    problem_traits traits;
    traits.is_least_squares           = false;
    traits.has_gradient               = problem.gradient.has_value();
    traits.has_hessian                = problem.hessian.has_value();
    traits.has_bounds                 = !problem.bounds.empty();
    traits.has_nonlinear_constraints  = !problem.constraints.empty();
    traits.num_parameters             = problem.num_parameters;
    return traits;
}

solver_result solve(
    const least_squares_problem& problem,
    const vector_type&           initial_guess,
    const solver_options&        options)
{
    if (problem.num_parameters == 0 || problem.num_residuals == 0 || !problem.residuals)
    {
        return failed(
            solver_status::invalid_problem,
            "least_squares_problem requires positive dimensions and residual callback",
            initial_guess);
    }

    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(
            solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }

    if (!problem.bounds.empty())
    {
        if ((problem.bounds.has_lower() &&
             problem.bounds.lower.size() != problem.num_parameters) ||
            (problem.bounds.has_upper() &&
             problem.bounds.upper.size() != problem.num_parameters))
        {
            return failed(
                solver_status::invalid_problem,
                "bounds size does not match num_parameters",
                initial_guess);
        }
    }

    auto chosen_algorithm = resolve_algorithm(options);
    auto solver_type = options.solver();

    switch (solver_type)
    {
    case solver_enum::LM:
    case solver_enum::GAUSS_NEWTON:
    case solver_enum::LBFGS:
    {
        vector_type x = initial_guess;
        solver_result result;
        result.parameters = x;
        result.backend    = backend::native;
        result.algorithm  = chosen_algorithm;

        jacobian_type jacobian = problem.jacobian ? *problem.jacobian : nullptr;

        switch (chosen_algorithm)
        {
        case algorithm::levenberg_marquardt:
        {
            levenberg_marquardt_solver solver(
                problem.num_parameters, problem.num_residuals, problem.residuals, jacobian);
            const auto out = solver.solve(x, static_cast<const solver_options_lm&>(options));

            result.status         = translate_native(out.status);
            result.parameters     = x;
            result.residual_norm  = out.residual_norm;
            result.objective      = 0.5 * out.residual_norm * out.residual_norm;
            result.iterations     = out.iterations;
            result.gradient_norm  = out.gradient_norm;
            result.step_norm      = out.step_norm;
            result.accepted_steps = out.accepted_steps;
            result.rejected_steps = out.rejected_steps;
            result.message        = out.message.empty() ? "converged" : out.message;
            return result;
        }

        case algorithm::gauss_newton:
        {
            gauss_newton_solver solver(
                problem.num_parameters, problem.num_residuals, problem.residuals, jacobian);
            const auto out = solver.solve(x, static_cast<const solver_options_gn&>(options));

            result.status         = translate_native(out.status);
            result.parameters     = x;
            result.residual_norm  = out.residual_norm;
            result.objective      = 0.5 * out.residual_norm * out.residual_norm;
            result.iterations     = out.iterations;
            result.gradient_norm  = out.gradient_norm;
            result.step_norm      = out.step_norm;
            result.accepted_steps = out.accepted_steps;
            result.rejected_steps = out.rejected_steps;
            result.message        = out.message.empty() ? "converged" : out.message;
            return result;
        }

        case algorithm::riemann_normal_coordinate_lm:
        {
            return solve_rnc_lm(
                problem, initial_guess, static_cast<const solver_options_rnc_lm&>(options));
        }

        default:
            result.status  = solver_status::unsupported_capability;
            result.message = "algorithm not supported by native backend";
            return result;
        }
    }
    case solver_enum::CERES:
    {
        if (!ceres_solver::is_supported())
        {
            return failed(solver_status::backend_unavailable, "Ceres backend not compiled in", initial_guess);
        }
        ceres_solver solver(problem.num_parameters, problem.num_residuals, problem.residuals,
                            problem.jacobian ? *problem.jacobian : nullptr,
                            problem.bounds.lower, problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        const bool          usable = solver.solve(x, static_cast<const solver_options_ceres&>(options));

        solver_result result;
        result.parameters = to_vector(x);
        result.status     = usable ? solver_status::converged : solver_status::numerical_failure;
        result.message    = usable ? "Ceres reported a usable solution" : "Ceres solution not usable";
        result.backend    = backend::ceres;
        vector_type residual(problem.num_residuals);
        problem.residuals(result.parameters, residual);
        result.objective = .5 * residual.squaredNorm();
        return result;
    }
    case solver_enum::IPOPT:
    {
        if (!ipopt_solver::is_supported())
        {
            return failed(solver_status::backend_unavailable, "Ipopt backend not compiled in", initial_guess);
        }
        if (!problem.jacobian)
        {
            return failed(solver_status::unsupported_capability,
                         "IPOPT least-squares requires a Jacobian callback",
                         initial_guess);
        }
        // Scalar view of F(x) = 0.5 * ||r(x)||^2 with gradient J^T r.
        const auto objective = [&problem](const vector_type& x)
        {
            vector_type residual(problem.num_residuals);
            problem.residuals(x, residual);
            return .5 * residual.squaredNorm();
        };
        const auto gradient = [&problem](const vector_type& x, vector_type& g)
        {
            vector_type residual(problem.num_residuals);
            problem.residuals(x, residual);
            matrix_type jacobian(problem.num_residuals, problem.num_parameters);
            (*problem.jacobian)(x, jacobian);
            g = jacobian.transpose() * residual;
        };
        ipopt_solver solver(problem.num_parameters, objective, gradient, nullptr,
                            problem.bounds.lower, problem.bounds.upper);
        std::vector<double> x = to_std(initial_guess);
        solver_result result  = from_backend(
            solver.solve_with_status(x, static_cast<const solver_options_ipopt&>(options)),
            x,
            backend::ipopt);
        result.objective = objective(result.parameters);
        return result;
    }
    case solver_enum::PETSC_TAO:
    {
        if (!petsc_tao_solver::is_supported())
        {
            return failed(solver_status::backend_unavailable, "PETSc/TAO backend not compiled in", initial_guess);
        }
        petsc_tao_solver solver(problem.num_parameters, problem.num_residuals, problem.residuals,
                                problem.jacobian ? *problem.jacobian : nullptr,
                                problem.bounds.lower, problem.bounds.upper);
        std::vector<double> x = to_std(initial_guess);
        solver_result result  = from_backend(
            solver.solve_with_status(x, static_cast<const solver_options_petsc&>(options)),
            x,
            backend::petsc_tao);
        vector_type residual(problem.num_residuals);
        problem.residuals(result.parameters, residual);
        result.objective = .5 * residual.squaredNorm();
        return result;
    }
    default:
        return failed(solver_status::backend_unavailable,
                     "unsupported solver backend",
                     initial_guess);
    }
}

solver_result solve(
    const optimization_problem& problem,
    const vector_type&          initial_guess,
    const solver_options&       options)
{
    if (problem.num_parameters == 0 || !problem.objective)
    {
        return failed(
            solver_status::invalid_problem,
            "optimization_problem requires positive dimensions and objective callback",
            initial_guess);
    }

    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(
            solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }

    if (!problem.bounds.empty())
    {
        if ((problem.bounds.has_lower() &&
             problem.bounds.lower.size() != problem.num_parameters) ||
            (problem.bounds.has_upper() &&
             problem.bounds.upper.size() != problem.num_parameters))
        {
            return failed(
                solver_status::invalid_problem,
                "bounds size does not match num_parameters",
                initial_guess);
        }
    }

    if (!problem.constraints.empty())
    {
        return failed(
            solver_status::unsupported_capability,
            "nonlinear constraints not yet implemented",
            initial_guess);
    }

    auto solver_type = options.solver();

    switch (solver_type)
    {
    case solver_enum::LBFGS:
    {
        vector_type x = initial_guess;

        solver_result result;
        result.parameters = x;
        result.backend    = backend::native;
        result.algorithm  = algorithm::lbfgs;

        gradient_type gradient = problem.gradient ? *problem.gradient : nullptr;
        lbfgs_solver  solver(problem.num_parameters, problem.objective, gradient);
        const auto    out = solver.solve(x, static_cast<const solver_options_bfgs&>(options));

        result.status         = translate_native(out.status);
        result.iterations     = out.iterations;
        result.parameters     = x;
        result.objective      = problem.objective(x);
        result.gradient_norm  = out.gradient_norm;
        result.step_norm      = out.step_norm;
        result.accepted_steps = out.accepted_steps;
        result.rejected_steps = out.rejected_steps;
        result.message        = out.message.empty() ? "converged" : out.message;
        return result;
    }
    case solver_enum::CERES:
    {
        return failed(solver_status::unsupported_capability,
                     "Ceres backend only supports least-squares problems",
                     initial_guess);
    }
    case solver_enum::IPOPT:
    {
        if (!ipopt_solver::is_supported())
        {
            return failed(solver_status::backend_unavailable, "Ipopt backend not compiled in", initial_guess);
        }
        if (!problem.gradient)
        {
            return failed(solver_status::unsupported_capability,
                         "IPOPT optimization requires a gradient callback",
                         initial_guess);
        }
        ipopt_solver solver(problem.num_parameters, problem.objective, *problem.gradient,
                            problem.hessian ? *problem.hessian : nullptr,
                            problem.bounds.lower, problem.bounds.upper);
        std::vector<double> x = to_std(initial_guess);
        solver_result result  = from_backend(
            solver.solve_with_status(x, static_cast<const solver_options_ipopt&>(options)),
            x,
            backend::ipopt);
        result.objective = problem.objective(result.parameters);
        return result;
    }
    case solver_enum::PETSC_TAO:
    {
        if (!petsc_tao_solver::is_supported())
        {
            return failed(solver_status::backend_unavailable, "PETSc/TAO backend not compiled in", initial_guess);
        }
        if (!problem.gradient)
        {
            return failed(solver_status::unsupported_capability,
                         "PETSc/TAO optimization requires a gradient callback",
                         initial_guess);
        }
        petsc_tao_solver solver(problem.num_parameters, problem.objective, *problem.gradient,
                                problem.hessian ? *problem.hessian : nullptr,
                                problem.bounds.lower, problem.bounds.upper);
        std::vector<double> x = to_std(initial_guess);
        solver_result result  = from_backend(
            solver.solve_with_status(x, static_cast<const solver_options_petsc&>(options)),
            x,
            backend::petsc_tao);
        result.objective = problem.objective(result.parameters);
        return result;
    }
    default:
        return failed(solver_status::backend_unavailable,
                     "unsupported solver backend",
                     initial_guess);
    }
}

}
