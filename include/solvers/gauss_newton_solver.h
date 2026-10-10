#ifndef SOLVERS_GAUSS_NEWTON_SOLVER_H_
#define SOLVERS_GAUSS_NEWTON_SOLVER_H_

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "detail/native_result.h"
#include "detail/support.h"

namespace solverslib
{
class solver_options_gn;

/**
 * @brief Plain (undamped) Gauss-Newton solver for nonlinear least squares.
 *
 * Solves the normal equations (J^T J) * step = J^T * r at each iteration -
 * unlike levenberg_marquardt_solver, with no damping/trust-region term - and
 * globalizes convergence with an Armijo backtracking line search along the
 * Gauss-Newton direction. That direction is always a descent direction for
 * 0.5*||r(x)||^2 whenever J has full column rank (step^T * gradient =
 * step^T * (J^T J) * step >= 0), so the line search is guaranteed to find an
 * accepted step unless J is (numerically) rank-deficient.
 *
 * Prefer levenberg_marquardt_solver when the Jacobian may be
 * ill-conditioned or the initial guess is far from the solution; Gauss-
 * Newton converges faster (locally quadratic, same as LM near the optimum)
 * but is less robust far from it.
 */
class gauss_newton_solver
{
    using scalar_type = double;

    function_type function_;
    jacobian_type jacobian_;
    size_t        num_parameters_;
    size_t        num_residuals_;

    // Step and scale are fixed at construction. Scratch is mutable because solve() is const.
    const double                   fd_step_;
    const finite_difference_scale  fd_scale_;
    mutable vector_type fd_parameters_;
    mutable vector_type fd_residual_;
    mutable vector_type fd_base_;

public:
    SOLVER_API gauss_newton_solver(size_t num_parameters,
        size_t                            num_residuals,
        function_type                     function,
        jacobian_type                     jacobian = nullptr,
        double                            bump = 1e-5,
        finite_difference_scale           difference_scale = finite_difference_scale::absolute);

    SOLVER_API native_result solve(vector_type& parameters, const solver_options_gn& options) const;
};
}  // namespace solverslib

#endif  // SOLVERS_GAUSS_NEWTON_SOLVER_H_
