#pragma once


#include <stdio.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "include/detail/support.h"
#include "include/detail/support.h"

namespace solverslib
{
enum class optimization_algorithm_convergence_type : int
{
    GRADIENT_CONVERGED   = 0,
    PARAMETERS_CONVERGED = 1,
    X2_CONVERGED         = 2,
    NOT_CONVERGED        = 3
};

struct optimization_algorithm_output
{
    using scalar_type = double;
    using size_type   = size_t;

    MATH_API explicit optimization_algorithm_output(size_type m);

    void update(
        bool                       x2_converged,
        bool                       parameters_converged,
        bool                       gradient_converged,
        size_type                  iteration,
        const Eigen::VectorXd& y_p);

    MATH_API void print() const;

    scalar_type                             x2_;
    std::vector<scalar_type>                errors_;
    size_type                               iterations_;
    optimization_algorithm_convergence_type status_;
};

}  // namespace solverslib
