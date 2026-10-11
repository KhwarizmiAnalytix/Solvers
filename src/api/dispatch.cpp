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
#include "solver_options/solver_options_nlopt.h"
#include "solver_options/solver_options_petsc.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/gauss_newton_solver.h"
#include "solvers/lbfgs_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/rnc_lm_solver.h"

#if SOLVERS_HAS_CERES
#include "solvers/ceres_solver.h"
#endif
#if SOLVERS_HAS_IPOPT
#include "solvers/ipopt_solver.h"
#endif
#if SOLVERS_HAS_NLOPT
#include "solvers/nlopt_solver.h"
#endif
#if SOLVERS_HAS_PETSC
#include "solvers/petsc_tao_solver.h"
#endif

// api::solve is the only call into a solver. Each overload checks the problem,
// selects a kernel from the concrete options type, and copies that kernel's
// outcome into solver_result.
//
// options.solver() names the library. Among native options the dynamic type
// names the algorithm: solver_options_lm runs Levenberg-Marquardt,
// solver_options_rnc_lm runs RNC-LM, solver_options_gn runs Gauss-Newton, and
// solver_options_bfgs runs L-BFGS. Ceres, Ipopt, NLopt, and PETSc/TAO are
// selected by solver() alone.
//
// A rejected request returns before any callback runs, with parameters left
// equal to the caller's guess. A compiled-out library returns
// backend_unavailable. A library that is present but cannot honor the request
// (bounds on native LM, a missing Jacobian for Ipopt least squares, nonlinear
// constraints) returns unsupported_capability.

namespace solverslib::api
{
namespace
{
using jacobian_type = std::function<void(const vector_type&, matrix_type&)>;
using gradient_type = std::function<void(const vector_type&, vector_type&)>;

// Early rejection. parameters is the guess the caller passed in.
solver_result failed(solver_status status, const std::string& message, const vector_type& x)
{
    solver_result result;
    result.status     = status;
    result.parameters = x;
    result.message    = message;
    return result;
}

// Ipopt, NLopt, and PETSc/TAO take std::vector<double>. Native kernels take Eigen.
vector_type to_vector(const std::vector<double>& values)
{
    return Eigen::Map<const vector_type>(values.data(), static_cast<index_type>(values.size()));
}

std::vector<double> to_std(const vector_type& values)
{
    return std::vector<double>(values.data(), values.data() + values.size());
}

// External adapters keep budget, stall, infeasibility, and a caller stop apart,
// and from_backend copies the library's own status code into backend_status.
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

solver_result from_backend(
    const backend_solve_status& out, const std::vector<double>& x, backend which)
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

// solver_options_rnc_lm derives from solver_options_lm, so both report
// solver() == LM. The dynamic_cast is what sends that object to RNC-LM.
// External options return automatic; their solve branches switch on solver().
algorithm resolve_algorithm(const solver_options& options)
{
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
}  // namespace

// Least squares. Native LM and Gauss-Newton store objective as ||r||^2 and
// residual_norm as ||r||. Ceres, Ipopt, NLopt, and PETSc/TAO store the scalar
// cost 0.5 * ||r||^2. RNC-LM builds its own solver_result, including the
// evaluation counters the other kernels leave unset.
solver_result solve(const least_squares_problem& problem,
    const vector_type&                           initial_guess,
    const solver_options&                        options)
{
    if (problem.num_parameters == 0 || problem.num_residuals == 0 || !problem.residuals)
    {
        return failed(solver_status::invalid_problem,
            "least_squares_problem requires positive dimensions and residual callback",
            initial_guess);
    }

    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }

    if (!problem.bounds.empty())
    {
        if ((problem.bounds.has_lower() && problem.bounds.lower.size() != problem.num_parameters) ||
            (problem.bounds.has_upper() && problem.bounds.upper.size() != problem.num_parameters))
        {
            return failed(solver_status::invalid_problem,
                "bounds size does not match num_parameters",
                initial_guess);
        }
    }

    auto chosen_algorithm = resolve_algorithm(options);
    auto solver_type      = options.solver();

    switch (solver_type)
    {
    case solver_enum::LM:
    case solver_enum::GAUSS_NEWTON:
    case solver_enum::LBFGS:
    {
        vector_type x = initial_guess;

        jacobian_type jacobian = problem.jacobian ? *problem.jacobian : nullptr;

        switch (chosen_algorithm)
        {
        case algorithm::levenberg_marquardt:
        {
            if (!problem.bounds.empty())
            {
                return failed(solver_status::unsupported_capability,
                    "native Levenberg-Marquardt does not support bounds",
                    initial_guess);
            }
            const auto&                lm_options = static_cast<const solver_options_lm&>(options);
            levenberg_marquardt_solver solver(problem.num_parameters,
                problem.num_residuals,
                problem.residuals,
                jacobian,
                lm_options.finite_difference_step(),
                lm_options.difference_scale());
            return solver.solve(x, lm_options);
        }

        case algorithm::gauss_newton:
        {
            // The kernel takes no bound vectors, so a bounds field is unused.
            const auto&         gn_options = static_cast<const solver_options_gn&>(options);
            gauss_newton_solver solver(problem.num_parameters,
                problem.num_residuals,
                problem.residuals,
                jacobian,
                gn_options.bump(),
                gn_options.difference_scale());
            return solver.solve(x, gn_options);
        }

        case algorithm::riemann_normal_coordinate_lm:
        {
            // The RNC kernel validates its own options and curve derivatives.
            return rnc_lm_solver::run(
                problem, initial_guess, static_cast<const solver_options_rnc_lm&>(options));
        }

        default:
            // L-BFGS options reach this native block and have no residual kernel.
            return failed(solver_status::unsupported_capability,
                "algorithm not supported by native backend",
                initial_guess);
        }
    }
    case solver_enum::CERES:
    {
#if SOLVERS_HAS_CERES
        ceres_solver        solver(problem.num_parameters,
            problem.num_residuals,
            problem.residuals,
            problem.jacobian ? *problem.jacobian : nullptr,
            problem.bounds.lower,
            problem.bounds.upper);
        std::vector<double> x = to_std(initial_guess);
        // Ceres reports success as a bool, so a budget stop and a numerical
        // failure both become numerical_failure. Iterations are not copied.
        const bool usable     = solver.solve(x, static_cast<const solver_options_ceres&>(options));

        solver_result result;
        result.parameters = to_vector(x);
        result.status     = usable ? solver_status::converged : solver_status::numerical_failure;
        result.message = usable ? "Ceres reported a usable solution" : "Ceres solution not usable";
        result.backend = backend::ceres;
        vector_type residual(problem.num_residuals);
        problem.residuals(result.parameters, residual);
        result.objective = .5 * residual.squaredNorm();
        return result;
#else
        return failed(
            solver_status::backend_unavailable, "Ceres backend not compiled in", initial_guess);
#endif
    }
    case solver_enum::IPOPT:
    {
#if SOLVERS_HAS_IPOPT
        if (!problem.jacobian)
        {
            return failed(solver_status::unsupported_capability,
                "IPOPT least-squares requires a Jacobian callback",
                initial_guess);
        }
        // Scalar view of F(x) = 0.5 * ||r(x)||^2 with gradient J^T r. Ipopt
        // has no residual interface, so both are rebuilt from the callbacks.
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
        ipopt_solver        solver(problem.num_parameters,
            objective,
            gradient,
            nullptr,
            problem.bounds.lower,
            problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        solver_result       result = from_backend(
            solver.solve(x, static_cast<const solver_options_ipopt&>(options)),
            x,
            backend::ipopt);
        result.objective = objective(result.parameters);
        return result;
#else
        return failed(
            solver_status::backend_unavailable, "Ipopt backend not compiled in", initial_guess);
#endif
    }
    case solver_enum::NLOPT:
    {
#if SOLVERS_HAS_NLOPT
        // Same scalar cost as Ipopt. The gradient callback is present only when
        // the problem supplies a Jacobian; derivative-free NLopt algorithms run
        // with it empty.
        const auto objective = [&problem](const vector_type& x)
        {
            vector_type residual(problem.num_residuals);
            problem.residuals(x, residual);
            return .5 * residual.squaredNorm();
        };
        const auto gradient = problem.jacobian
            ? std::function<void(const vector_type&, vector_type&)>(
                  [&problem](const vector_type& x, vector_type& g)
                  {
                      vector_type residual(problem.num_residuals);
                      problem.residuals(x, residual);
                      matrix_type jacobian(problem.num_residuals, problem.num_parameters);
                      (*problem.jacobian)(x, jacobian);
                      g = jacobian.transpose() * residual;
                  })
            : nullptr;
        nlopt_solver        solver(
            problem.num_parameters, objective, gradient, problem.bounds.lower, problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        solver_result       result = from_backend(
            solver.solve(x, static_cast<const solver_options_nlopt&>(options)),
            x,
            backend::nlopt);
        result.objective = objective(result.parameters);
        return result;
#else
        return failed(
            solver_status::backend_unavailable, "NLopt backend not compiled in", initial_guess);
#endif
    }
    case solver_enum::PETSC_TAO:
    {
#if SOLVERS_HAS_PETSC
        // TAO receives the residual callback directly and chooses its algorithm
        // from solver_options_petsc (POUNDERS, BRGN, NLS, and the rest).
        petsc_tao_solver    solver(problem.num_parameters,
            problem.num_residuals,
            problem.residuals,
            problem.jacobian ? *problem.jacobian : nullptr,
            problem.bounds.lower,
            problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        solver_result       result = from_backend(
            solver.solve(x, static_cast<const solver_options_petsc&>(options)),
            x,
            backend::petsc_tao);
        vector_type residual(problem.num_residuals);
        problem.residuals(result.parameters, residual);
        result.objective = .5 * residual.squaredNorm();
        return result;
#else
        return failed(solver_status::backend_unavailable,
            "PETSc/TAO backend not compiled in",
            initial_guess);
#endif
    }
    default:
        return failed(
            solver_status::backend_unavailable, "unsupported solver backend", initial_guess);
    }
}

// Scalar optimization. Native L-BFGS is the only native kernel. Ceres is
// least-squares only. Ipopt, NLopt, and PETSc/TAO each require a gradient
// callback. Nonlinear constraints are rejected; the constraints struct carries
// counts and no callback for a kernel to evaluate.
solver_result solve(const optimization_problem& problem,
    const vector_type&                          initial_guess,
    const solver_options&                       options)
{
    if (problem.num_parameters == 0 || !problem.objective)
    {
        return failed(solver_status::invalid_problem,
            "optimization_problem requires positive dimensions and objective callback",
            initial_guess);
    }

    if (static_cast<std::size_t>(initial_guess.size()) != problem.num_parameters)
    {
        return failed(solver_status::invalid_problem,
            "initial_guess size does not match num_parameters",
            initial_guess);
    }

    if (!problem.bounds.empty())
    {
        if ((problem.bounds.has_lower() && problem.bounds.lower.size() != problem.num_parameters) ||
            (problem.bounds.has_upper() && problem.bounds.upper.size() != problem.num_parameters))
        {
            return failed(solver_status::invalid_problem,
                "bounds size does not match num_parameters",
                initial_guess);
        }
    }

    if (!problem.constraints.empty())
    {
        return failed(solver_status::unsupported_capability,
            "nonlinear constraints not yet implemented",
            initial_guess);
    }

    auto solver_type = options.solver();

    switch (solver_type)
    {
    case solver_enum::LBFGS:
    {
        vector_type x = initial_guess;

        gradient_type gradient     = problem.gradient ? *problem.gradient : nullptr;
        const auto&   bfgs_options = static_cast<const solver_options_bfgs&>(options);
        lbfgs_solver  solver(problem.num_parameters,
            problem.objective,
            gradient,
            bfgs_options.bump(),
            bfgs_options.difference_scale());
        return solver.solve(x, bfgs_options);
    }
    case solver_enum::CERES:
    {
        return failed(solver_status::unsupported_capability,
            "Ceres backend only supports least-squares problems",
            initial_guess);
    }
    case solver_enum::IPOPT:
    {
#if SOLVERS_HAS_IPOPT
        if (!problem.gradient)
        {
            return failed(solver_status::unsupported_capability,
                "IPOPT optimization requires a gradient callback",
                initial_guess);
        }
        ipopt_solver        solver(problem.num_parameters,
            problem.objective,
            *problem.gradient,
            problem.hessian ? *problem.hessian : nullptr,
            problem.bounds.lower,
            problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        solver_result       result = from_backend(
            solver.solve(x, static_cast<const solver_options_ipopt&>(options)),
            x,
            backend::ipopt);
        result.objective = problem.objective(result.parameters);
        return result;
#else
        return failed(
            solver_status::backend_unavailable, "Ipopt backend not compiled in", initial_guess);
#endif
    }
    case solver_enum::NLOPT:
    {
#if SOLVERS_HAS_NLOPT
        if (!problem.gradient)
        {
            return failed(solver_status::unsupported_capability,
                "NLopt optimization requires a gradient callback",
                initial_guess);
        }
        nlopt_solver        solver(problem.num_parameters,
            problem.objective,
            *problem.gradient,
            problem.bounds.lower,
            problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        solver_result       result = from_backend(
            solver.solve(x, static_cast<const solver_options_nlopt&>(options)),
            x,
            backend::nlopt);
        result.objective = problem.objective(result.parameters);
        return result;
#else
        return failed(
            solver_status::backend_unavailable, "NLopt backend not compiled in", initial_guess);
#endif
    }
    case solver_enum::PETSC_TAO:
    {
#if SOLVERS_HAS_PETSC
        if (!problem.gradient)
        {
            return failed(solver_status::unsupported_capability,
                "PETSc/TAO optimization requires a gradient callback",
                initial_guess);
        }
        petsc_tao_solver    solver(problem.num_parameters,
            problem.objective,
            *problem.gradient,
            problem.hessian ? *problem.hessian : nullptr,
            problem.bounds.lower,
            problem.bounds.upper);
        std::vector<double> x      = to_std(initial_guess);
        solver_result       result = from_backend(
            solver.solve(x, static_cast<const solver_options_petsc&>(options)),
            x,
            backend::petsc_tao);
        result.objective = problem.objective(result.parameters);
        return result;
#else
        return failed(solver_status::backend_unavailable,
            "PETSc/TAO backend not compiled in",
            initial_guess);
#endif
    }
    default:
        // Native LM, Gauss-Newton, and RNC-LM options have no scalar kernel.
        return failed(
            solver_status::backend_unavailable, "unsupported solver backend", initial_guess);
    }
}

}  // namespace solverslib::api
