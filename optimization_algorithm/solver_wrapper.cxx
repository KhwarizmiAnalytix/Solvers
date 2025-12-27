#include "optimization_algorithm/solver_wrapper.h"

#include "common/pointer.h"
#include "optimization_algorithm/solver_options/solver_options.h"
#include "optimization_algorithm/solver_options/solver_options_bfgs.h"
#include "optimization_algorithm/solver_options/solver_options_ceres.h"
#include "optimization_algorithm/solver_options/solver_options_lm.h"
#include "optimization_algorithm/solver_options/solver_options_nlopt.h"
#include "optimization_algorithm/solver_output.h"
#include "optimization_algorithm/solvers/ceres_solver.h"
#include "optimization_algorithm/solvers/lbfgs_solver.h"
#include "optimization_algorithm/solvers/levenberg_marquardt_solver.h"
#include "optimization_algorithm/solvers/nlopt_solver.h"
#include "util/logger.h"

namespace quarisma
{

solver_wrapper::solver_wrapper(
    size_t                      num_parameters,
    size_t                      num_residuals,
    objective_function_type     objective_function,
    objective_function_aad_type objective_function_aad,
    const std::vector<double>&  lower_bounds,
    const std::vector<double>&  upper_bounds)
    : num_parameters_(num_parameters),
      num_residuals_(num_residuals),
      objective_function_(std::move(objective_function)),
      objective_function_aad_(std::move(objective_function_aad)),
      lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds)
{
    QUARISMA_CHECK(num_parameters > 0, "Number of parameters must be positive");
    QUARISMA_CHECK(num_residuals > 0, "Number of residuals must be positive");
    QUARISMA_CHECK(objective_function_ != nullptr, "Objective function cannot be null");

    // Validate bounds if provided
    if (!lower_bounds_.empty())
    {
        QUARISMA_CHECK(
            lower_bounds_.size() == num_parameters_,
            "Lower bounds size must match number of parameters");
    }
    if (!upper_bounds_.empty())
    {
        QUARISMA_CHECK(
            upper_bounds_.size() == num_parameters_,
            "Upper bounds size must match number of parameters");
    }
}

bool solver_wrapper::solve(
    std::vector<double>& parameters, const ptr_const<solver_options>& options) const
{
    QUARISMA_CHECK(
        parameters.size() == num_parameters_,
        "Parameter vector size must match number of parameters");
    QUARISMA_CHECK(options != nullptr, "Solver options cannot be null");

    if (const auto& ceres_options = std::dynamic_pointer_cast<const solver_options_ceres>(options))
    {
        return solve_ceres(parameters, *ceres_options);
    }

    if (const auto& lm_options = std::dynamic_pointer_cast<const solver_options_lm>(options))
    {
        return solve_lm(parameters, *lm_options);
    }

    if (const auto& nlopt_options = std::dynamic_pointer_cast<const solver_options_nlopt>(options))
    {
        return solve_nlopt(parameters, *nlopt_options);
    }

    if (const auto& bfgs_options = std::dynamic_pointer_cast<const solver_options_bfgs>(options))
    {
        return solve_lbfgs(parameters, *bfgs_options);
    }

    QUARISMA_THROW("Unsupported solver_options");
}

bool solver_wrapper::solve_ceres(
    std::vector<double>& parameters, const solver_options_ceres& options) const
{
    if (!ceres_solver::is_supported())
    {
        return false;
    }

    const auto& solver = std::make_unique<ceres_solver>(
        num_parameters_,
        num_residuals_,
        objective_function_,
        options.aad_jacobian() ? objective_function_aad_ : nullptr,
        lower_bounds_,
        upper_bounds_);

    return solver->solve(parameters, options);
}

bool solver_wrapper::solve_lm(
    std::vector<double>& parameters, const solver_options_lm& options) const
{
    // Convert std::vector to quarisma::vector for LM solver
    vector<double> quarisma_params(parameters.data(), parameters.size());

    levenberg_marquardt_solver solver(
        num_parameters_,
        num_residuals_,
        objective_function_,
        options.aad_jacobian() ? objective_function_aad_ : nullptr);

    auto result = solver.solve(quarisma_params, options);

    if (options.verbose())
    {
        result.print();
    }

    return result.status_ != solver_convergence_enum::NOT_CONVERGED;
}

bool solver_wrapper::solve_nlopt(
    std::vector<double>& parameters, const solver_options_nlopt& options) const
{
    if (!nlopt_solver::is_supported())
    {
        return false;
    }

    nlopt_solver solver(
        num_parameters_,
        num_residuals_,
        objective_function_,
        options.aad_jacobian() ? objective_function_aad_ : nullptr,
        lower_bounds_,
        upper_bounds_);

    solver.solve(parameters, options);

    return true;
}

bool solver_wrapper::solve_lbfgs(
    std::vector<double>& parameters, const solver_options_bfgs& options) const
{
    vector<double> quarisma_params(parameters.data(), parameters.size());

    const auto& solver = std::make_unique<lbfgs_solver>(
        num_parameters_,
        num_residuals_,
        objective_function_,
        options.aad_jacobian() ? objective_function_aad_ : nullptr);

    auto result = solver->solve(quarisma_params, options);

    if (options.verbose())
    {
        result.print();
    }

    return result.status_ != solver_convergence_enum::NOT_CONVERGED;
}

bool solver_wrapper::is_supported(solver_enum optimizer_type)
{
    switch (optimizer_type)
    {
    case solver_enum::CERES:
        return ceres_solver::is_supported();
    case solver_enum::LM:
        return true;
    case solver_enum::NLOPT:
        return nlopt_solver::is_supported();
    case solver_enum::LBFGS:
        return true;
    default:
        return false;
    }
}
}  // namespace quarisma
