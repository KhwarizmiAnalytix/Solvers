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

    // DEPRECATED: Use the provider-based constructor instead. The AAD path is
    // now configured via problem.derivatives(auto_diff()) or set_jacobian_provider().
    using CostFunctionLambda_aad = std::function<void(const vector_type&, matrix_type&)>;

    // DEPRECATED: Use ceres_solver(n, m, cost_fn, provider_factory, ...) instead.
    [[deprecated("Use ceres_solver(n, m, cost_fn, provider_factory, ...) instead")]]
    SOLVER_API ceres_solver(size_t num_parameters,
        size_t                     num_residuals,
        CostFunctionLambda         cost_function,
        CostFunctionLambda_aad     cost_function_aad = nullptr,
        const std::vector<double>& lower_bounds      = {},
        const std::vector<double>& upper_bounds      = {});

    // New constructor with provider support
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
    CostFunctionLambda     cost_function_;
    CostFunctionLambda_aad cost_function_aad_;

    std::shared_ptr<const api::detail::provider_factory> provider_;

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    size_t              num_parameters_;
    size_t              num_residuals_;
};
}  // namespace solverslib
