#pragma once
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <vector>

#include "include/detail/support.h"

namespace solverslib
{

class solver_options_ceres;

class ceres_solver_algorithms
{
public:
    using CostFunctionLambda     = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using CostFunctionLambda_aad = std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)>;

    SOLVER_API ceres_solver_algorithms(int num_parameters,
        int                                num_residuals,
        CostFunctionLambda                 cost_function,
        CostFunctionLambda_aad             cost_function_aad,
        const std::vector<double>&         lower_bounds,
        const std::vector<double>&         upper_bounds);

    SOLVER_API ceres_solver_algorithms(int num_parameters,
        int                                num_residuals,
        CostFunctionLambda                 cost_function,
        const std::vector<double>&         lower_bounds,
        const std::vector<double>&         upper_bounds);

    SOLVER_API bool solve(std::vector<double>& parameters, const solver_options_ceres& option);

    SOLVER_API static bool is_supported();

private:
    CostFunctionLambda     cost_function_;
    CostFunctionLambda_aad cost_function_aad_;

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    int                 num_parameters_;
    int                 num_residuals_;
};
}  // namespace solverslib
