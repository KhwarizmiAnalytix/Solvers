#pragma once

//#ifndef __QUARISMA_WRAP__

#include <stdio.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "MathModule.h"
#include "expressions/expressions.h"
#include "matrix_operation/linear_solver.h"
#include "memory/allocator.h"
#include "optimization_algorithm/optimization_algorithm_output.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"

namespace quarisma
{
class solver_options_lm;

class levenberg_marquardt
{
    using scalar_type = double;
    using vector_type = vector<double>;
    using matrix_type = matrix<double>;

    using function_type = std::function<void(vector_type const&, vector_type&)>;
    using jacobian_type = std::function<void(vector_type const&, matrix_type&)>;

    function_type function_;
    jacobian_type jacobian_;
    int           num_parameters_;
    int           num_residuals_;

public:
    MATH_API levenberg_marquardt(
        int num_parameters, int num_residuals, function_type function, jacobian_type jacobian);

    MATH_API levenberg_marquardt(int num_parameters, int num_residuals, function_type function);

    MATH_API optimization_algorithm_output
    solve(vector<double>& parameters, const solver_options_lm& options) const;
};
}  // namespace quarisma
//#endif
