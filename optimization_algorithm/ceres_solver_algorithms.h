#include <functional>
#include <iostream>
#include <vector>

#include "MathModule.h"

namespace quarisma
{

template <typename T>
class vector;

template <typename T>
class matrix;

class solver_options_ceres;

class ceres_solver_algorithms
{
public:
    using CostFunctionLambda     = std::function<void(const vector<double>&, vector<double>&)>;
    using CostFunctionLambda_aad = std::function<void(const vector<double>&, matrix<double>&)>;

    MATH_API ceres_solver_algorithms(
        int                        num_parameters,
        int                        num_residuals,
        CostFunctionLambda         cost_function,
        CostFunctionLambda_aad     cost_function_aad,
        const std::vector<double>& lower_bounds,
        const std::vector<double>& upper_bounds);

    MATH_API ceres_solver_algorithms(
        int                        num_parameters,
        int                        num_residuals,
        CostFunctionLambda         cost_function,
        const std::vector<double>& lower_bounds,
        const std::vector<double>& upper_bounds);

    MATH_API bool solve(std::vector<double>& parameters, const solver_options_ceres& option);

    MATH_API static bool is_supported();

private:
    CostFunctionLambda     cost_function_;
    CostFunctionLambda_aad cost_function_aad_;

    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    int                 num_parameters_;
    int                 num_residuals_;
};
}  // namespace quarisma