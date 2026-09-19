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
enum class solver_convergence_enum : int
{
    GRADIENT_CONVERGED   = 0,
    PARAMETERS_CONVERGED = 1,
    X2_CONVERGED         = 2,
    NOT_CONVERGED        = 3
};

struct solver_output
{
    using scalar_type = double;
    using size_type   = size_t;

    SOLVER_API explicit solver_output(size_type m);

    void update(
        bool                       x2_converged,
        bool                       parameters_converged,
        bool                       gradient_converged,
        size_type                  iteration,
        const Eigen::VectorXd& y_p);

    SOLVER_API void print() const;

    scalar_type              x2_;
    std::vector<scalar_type> errors_;
    size_type                iterations_;
    solver_convergence_enum  status_;
};

}  // namespace solverslib
