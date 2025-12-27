#pragma once

#ifndef __QUARISMA_WRAP__

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "MathModule.h"
#include "expressions/expressions.h"
#include "optimization_algorithm/optimization_algorithm_output.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"
#include "util/exception.h"

namespace quarisma
{
class solver_options_bfgs;
class LBFGS
{
    using size_type   = size_t;
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
    MATH_API LBFGS(
        int num_parameters, int num_residuals, function_type function, jacobian_type jacobian);

    MATH_API LBFGS(int num_parameters, int num_residuals, function_type function);

    MATH_API optimization_algorithm_output
    solve(vector<double>& parameters, const solver_options_bfgs& options) const;
};
}  // namespace quarisma
#endif
