#ifndef SOLVERS_LEVENBERG_MARQUARDT_SOLVER_H_
#define SOLVERS_LEVENBERG_MARQUARDT_SOLVER_H_

#include <stdio.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "detail/native_result.h"
#include "detail/support.h"

namespace solverslib
{
class solver_options_lm;

class levenberg_marquardt_solver
{
    using scalar_type = double;

    function_type function_;
    jacobian_type jacobian_;
    size_t        num_parameters_;
    size_t        num_residuals_;

    // Step and scale are fixed at construction. Scratch is mutable because solve() is const.
    const double                  fd_step_;
    const finite_difference_scale fd_scale_;
    mutable vector_type           fd_parameters_;
    mutable vector_type           fd_residual_;
    mutable vector_type           fd_base_;

public:
    SOLVER_API levenberg_marquardt_solver(size_t num_parameters,
        size_t                                   num_residuals,
        function_type                            function,
        jacobian_type                            jacobian               = nullptr,
        double                                   finite_difference_step = 1e-5,
        finite_difference_scale difference_scale = finite_difference_scale::absolute);

    SOLVER_API native_result solve(vector_type& parameters, const solver_options_lm& options) const;
};
}  // namespace solverslib

#endif  // SOLVERS_LEVENBERG_MARQUARDT_SOLVER_H_
