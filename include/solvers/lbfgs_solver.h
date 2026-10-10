#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "detail/native_result.h"
#include "detail/support.h"

namespace solverslib
{
class solver_options_bfgs;
class lbfgs_solver
{
    using size_type   = size_t;
    using scalar_type = double;

    using objective_type = std::function<double(vector_type const&)>;
    using gradient_type  = std::function<void(vector_type const&, vector_type&)>;

    function_type function_;
    jacobian_type jacobian_;
    size_type     num_parameters_;
    size_type     num_residuals_;

    // Step and scale are fixed at construction. Scratch is mutable because run() is const.
    const double                  fd_step_;
    const finite_difference_scale fd_scale_;
    mutable vector_type           fd_parameters_;
    mutable vector_type           fd_residual_;
    mutable vector_type           fd_base_;

    objective_type objective_;
    gradient_type  gradient_;
    bool           scalar_mode_ = false;

    native_result run(vector_type& parameters, const solver_options_bfgs& options) const;

public:
    SOLVER_API lbfgs_solver(size_type num_parameters,
        size_type                     num_residuals,
        function_type                 function,
        jacobian_type                 jacobian         = nullptr,
        double                        bump             = 1e-6,
        finite_difference_scale       difference_scale = finite_difference_scale::absolute);

    SOLVER_API lbfgs_solver(size_type num_parameters,
        objective_type                objective,
        gradient_type                 gradient,
        double                        bump             = 1e-6,
        finite_difference_scale       difference_scale = finite_difference_scale::absolute);

    SOLVER_API native_result solve(
        vector_type& parameters, const solver_options_bfgs& options) const;
};
}  // namespace solverslib
