#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "include/detail/support.h"
#include "include/solver_output.h"

namespace solverslib
{
class solver_options_bfgs;
class lbfgs_solver
{
    using size_type   = size_t;
    using scalar_type = double;

    using function_type = std::function<void(vector_type const&, vector_type&)>;
    using jacobian_type = std::function<void(vector_type const&, matrix_type&)>;

    function_type function_;
    jacobian_type jacobian_;
    size_type     num_parameters_;
    size_type     num_residuals_;

public:
    SOLVER_API               lbfgs_solver(size_type num_parameters,
                      size_type                     num_residuals,
                      function_type                 function,
                      jacobian_type                 jacobian = nullptr);
    SOLVER_API solver_output solve(
        vector_type& parameters, const solver_options_bfgs& options) const;
};
}  // namespace solverslib
