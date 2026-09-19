#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "include/detail/support.h"
#include "include/detail/support.h"
#include "include/solver_enum.h"

namespace solverslib
{
class solver_options;
class solver_options_ceres;
class solver_options_lm;
class solver_options_nlopt;
class solver_options_bfgs;

/**
 * @brief Universal optimizer wrapper that automatically selects the appropriate optimizer
 * based on solver_options type. Eliminates the need for switch statements in calibration functions.
 *
 * This wrapper provides a unified interface for all optimization algorithms (Ceres, LM, NLopt, lbfgs_solver)
 * and automatically dispatches to the correct optimizer based on the solver_options type.
 *
 * Usage:
 * @code
 * // Define objective functions
 * auto objective_function = [&](const Eigen::VectorXd& x, Eigen::VectorXd& residuals) {
 *     // Your objective function implementation
 * };
 *
 * auto objective_function_aad = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) {
 *     // Your AAD jacobian implementation (optional)
 * };
 *
 * // Create wrapper and solve
 * solver_wrapper wrapper(num_parameters, num_residuals,
 *                          objective_function, objective_function_aad,
 *                          lower_bounds, upper_bounds);
 *
 * bool success = wrapper.solve(parameters, options);
 * @endcode
 */
class SOLVER_VISIBILITY solver_wrapper
{
public:
    // Function type definitions matching the optimizer interfaces
    using objective_function_type     = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using objective_function_aad_type = std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)>;

    /**
     * @brief Constructor with both objective function and AAD jacobian
     *
     * @param num_parameters Number of optimization parameters
     * @param num_residuals Number of residuals/observations
     * @param objective_function Main objective function
     * @param objective_function_aad AAD jacobian function (can be nullptr)
     * @param lower_bounds Lower bounds for parameters (optional)
     * @param upper_bounds Upper bounds for parameters (optional)
     */
    SOLVER_API solver_wrapper(
        size_t                      num_parameters,
        size_t                      num_residuals,
        objective_function_type     objective_function,
        objective_function_aad_type objective_function_aad = nullptr,
        const std::vector<double>&  lower_bounds           = {},
        const std::vector<double>&  upper_bounds           = {});

    /**
     * @brief Universal solve method that automatically dispatches to the correct optimizer
     *
     * @param parameters Input/output parameter vector
     * @param options Solver options (determines which optimizer to use)
     * @return true if optimization succeeded, false otherwise
     */
    SOLVER_API bool solve(
        std::vector<double>& parameters, const std::shared_ptr<const solver_options>& options) const;

    /**
     * @brief Check if the specified solver is supported
     *
     * @param solver_type The solver type to check
     * @return true if supported, false otherwise
     */
    SOLVER_API static bool is_supported(solver_enum solver_type);

private:
    // Internal solve methods for each optimizer type
    bool solve_ceres(std::vector<double>& parameters, const solver_options_ceres& options) const;
    bool solve_lm(std::vector<double>& parameters, const solver_options_lm& options) const;
    bool solve_nlopt(std::vector<double>& parameters, const solver_options_nlopt& options) const;
    bool solve_lbfgs(std::vector<double>& parameters, const solver_options_bfgs& options) const;

    // Member variables
    size_t num_parameters_;
    size_t num_residuals_;

    // Objective functions (one set will be used depending on constructor)
    objective_function_type     objective_function_;
    objective_function_aad_type objective_function_aad_;

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};
}  // namespace solverslib
