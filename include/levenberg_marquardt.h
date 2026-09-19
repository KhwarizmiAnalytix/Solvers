#pragma once


#include <stdio.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "include/detail/support.h"
#include "include/detail/support.h"
#include "include/optimization_algorithm_output.h"

namespace solverslib
{
class solver_options_lm;

class levenberg_marquardt
{
    using scalar_type = double;
    using vector_type = Eigen::VectorXd;
    using matrix_type = Eigen::MatrixXd;

    using function_type = std::function<void(vector_type const&, vector_type&)>;
    using jacobian_type = std::function<void(vector_type const&, matrix_type&)>;

    function_type function_;
    jacobian_type jacobian_;
    int           num_parameters_;
    int           num_residuals_;

public:
    SOLVER_API levenberg_marquardt(
        int num_parameters, int num_residuals, function_type function, jacobian_type jacobian);

    SOLVER_API levenberg_marquardt(int num_parameters, int num_residuals, function_type function);

    SOLVER_API optimization_algorithm_output
    solve(Eigen::VectorXd& parameters, const solver_options_lm& options) const;
};
}  // namespace solverslib
