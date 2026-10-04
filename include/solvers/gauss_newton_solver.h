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
namespace api::detail
{
class residual_evaluator;
}
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

    using function_type = std::function<void(vector_type const&, vector_type&)>;
    using jacobian_type = std::function<void(vector_type const&, matrix_type&)>;

    function_type function_;
    jacobian_type jacobian_;
    size_t        num_parameters_;
    size_t        num_residuals_;

    // Non-owning; when set, takes precedence over function_/jacobian_.
    api::detail::residual_evaluator* evaluator_ = nullptr;

public:
    SOLVER_API gauss_newton_solver(size_t num_parameters,
        size_t                            num_residuals,
        function_type                     function,
        jacobian_type                     jacobian = nullptr);

    // Evaluator-driven form; the evaluator must outlive the solver.
    SOLVER_API explicit gauss_newton_solver(api::detail::residual_evaluator& evaluator);

    SOLVER_API native_result solve(vector_type& parameters, const solver_options_gn& options) const;
};
}  // namespace solverslib

#endif  // SOLVERS_GAUSS_NEWTON_SOLVER_H_
