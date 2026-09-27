#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "detail/support.h"

namespace solverslib::api::detail
{
class provider_factory;
}

namespace solverslib
{

class solver_options_ceres;

class ceres_solver
{
public:
    using CostFunctionLambda = std::function<void(const vector_type&, vector_type&)>;

    // Constructor with explicit Jacobian callback.
    // Prefer using the provider-based constructor with set_jacobian_provider().
    SOLVER_API ceres_solver(size_t num_parameters,
        size_t                                                      num_residuals,
        CostFunctionLambda                                          cost_function,
        std::function<void(const vector_type&, matrix_type&)>       jacobian_callback,
        const std::vector<double>&                                  lower_bounds = {},
        const std::vector<double>&                                  upper_bounds = {});

    // Constructor with provider support
    SOLVER_API ceres_solver(size_t num_parameters,
        size_t                                                    num_residuals,
        CostFunctionLambda                                        cost_function,
        std::shared_ptr<const api::detail::provider_factory>     provider        = nullptr,
        const std::vector<double>&                                lower_bounds    = {},
        const std::vector<double>&                                upper_bounds    = {});

    SOLVER_API bool solve(std::vector<double>& parameters, const solver_options_ceres& option);

    // Solve and return Ceres summary for detailed diagnostics
    SOLVER_API void solve_with_summary(
        std::vector<double>&    parameters,
        const solver_options_ceres& options,
        void*                   summary_ptr);  // ceres::Solver::Summary*

    SOLVER_API static bool is_supported();

private:
    CostFunctionLambda                                    cost_function_;
    std::function<void(const vector_type&, matrix_type&)> jacobian_callback_;

    std::shared_ptr<const api::detail::provider_factory> provider_;

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    size_t              num_parameters_;
    size_t              num_residuals_;
};
}  // namespace solverslib
