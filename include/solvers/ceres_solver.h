#pragma once
#include <functional>
#include <vector>

#include "detail/support.h"

namespace solverslib
{

class solver_options_ceres;

class ceres_solver
{
public:
    // Constructor with explicit Jacobian callback.
    SOLVER_API ceres_solver(size_t num_parameters,
        size_t                              num_residuals,
        function_type                       cost_function,
        const jacobian_type&                jacobian_callback = nullptr,
        const std::vector<double>&          lower_bounds      = {},
        const std::vector<double>&          upper_bounds      = {});

    SOLVER_API bool solve(std::vector<double>& parameters, const solver_options_ceres& options);

    // Solve and return Ceres summary for detailed diagnostics
    SOLVER_API void solve_with_summary(
        std::vector<double>&           parameters,
        const solver_options_ceres&    options,
        void*                          summary_ptr);  // ceres::Solver::Summary*

    SOLVER_API static bool is_supported();

private:
    function_type              cost_function_;
    jacobian_type              jacobian_callback_;
    std::vector<double>        lower_bounds_;
    std::vector<double>        upper_bounds_;
    size_t                     num_parameters_;
    size_t                     num_residuals_;
};
}  // namespace solverslib
