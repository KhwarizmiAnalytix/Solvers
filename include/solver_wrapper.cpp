#include "solver_wrapper.h"

#include "detail/support.h"
#include "solver_options/solver_options.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_nlopt.h"
#include "solver_output.h"
#include "solvers/ceres_solver.h"
#include "solvers/lbfgs_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/nlopt_solver.h"

namespace solverslib
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
    SOLVERS_CHECK(num_parameters > 0, "Number of parameters must be positive");
    SOLVERS_CHECK(num_residuals > 0, "Number of residuals must be positive");
    SOLVERS_CHECK(objective_function_ != nullptr, "Objective function cannot be null");

    // Validate bounds if provided
    if (!lower_bounds_.empty())
    {
        SOLVERS_CHECK(
            lower_bounds_.size() == num_parameters_,
            "Lower bounds size must match number of parameters");
    }
    if (!upper_bounds_.empty())
    {
        SOLVERS_CHECK(
            upper_bounds_.size() == num_parameters_,
            "Upper bounds size must match number of parameters");
    }
}

bool solver_wrapper::solve(
    std::vector<double>& parameters, const std::shared_ptr<const solver_options>& options) const
{
    SOLVERS_CHECK(
        parameters.size() == num_parameters_,
        "Parameter vector size must match number of parameters");
    SOLVERS_CHECK(options != nullptr, "Solver options cannot be null");

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

    SOLVERS_THROW("Unsupported solver_options");
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
    // Convert std::vector to solverslib::vector for LM solver
    vector_type quarisma_params = Eigen::Map<const vector_type>(parameters.data(), parameters.size());

    levenberg_marquardt_solver solver(
        num_parameters_,
        num_residuals_,
        objective_function_,
        options.aad_jacobian() ? objective_function_aad_ : nullptr);

    auto result = solver.solve(quarisma_params, options);
    Eigen::Map<vector_type>(parameters.data(), parameters.size()) = quarisma_params;

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
    vector_type quarisma_params = Eigen::Map<const vector_type>(parameters.data(), parameters.size());

    const auto& solver = std::make_unique<lbfgs_solver>(
        num_parameters_,
        num_residuals_,
        objective_function_,
        options.aad_jacobian() ? objective_function_aad_ : nullptr);

    auto result = solver->solve(quarisma_params, options);
    Eigen::Map<vector_type>(parameters.data(), parameters.size()) = quarisma_params;

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
}  // namespace solverslib
