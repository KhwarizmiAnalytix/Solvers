#ifndef SOLVERS_STATUS_H_
#define SOLVERS_STATUS_H_

#include <cstdint>

#include "detail/support.h"

namespace solverslib::api
{
// One termination vocabulary shared across every backend. Adapters translate
// their native codes (Ceres, PETSc/TAO, Ipopt, or the native
// solver_convergence_enum) into exactly one of these. See the redesign review,
// "Return a structured result", for the rationale on keeping this closed set.
enum class solver_status : std::uint8_t
{
    converged,               // a named stopping criterion was met
    max_iterations,          // iteration/evaluation budget exhausted with a usable iterate
    invalid_problem,         // dimensions, callbacks, or options were rejected before solving
    numerical_failure,       // non-finite quantities or a failed linear solve during the run
    infeasible,              // a supplied point/constraint set is infeasible
    unsupported_capability,  // the chosen backend cannot honor a requested capability
    backend_unavailable,     // the requested backend was not compiled in
    user_stopped,            // a caller callback requested a stop
    stalled                  // no acceptable step exists; the last accepted iterate is valid
};

// Which stopping criterion a converged native run actually satisfied. Only
// native solvers (LM, Gauss-Newton, L-BFGS) distinguish these; external
// backends collapse to solver_status::converged without a reason, so this
// stays nullopt in solver_result for them.
enum class convergence_reason : std::uint8_t
{
    function,   // the objective/residual value met its tolerance
    parameter,  // the parameter step length met its tolerance
    gradient    // the gradient (or JtWr) norm met its tolerance
};

// Which algorithm was actually run. Kept independent of the backend that ran
// it (PyTorch/Eigen style: the mathematical method is named separately from the
// storage/execution backend that realizes it).
enum class algorithm : std::uint8_t
{
    automatic = 0,
    levenberg_marquardt,
    gauss_newton,
    lbfgs,
    riemann_normal_coordinate_lm
};

// Which library/executor realized the algorithm.
enum class backend : std::uint8_t
{
    automatic = 0,
    native,
    ipopt,
    nlopt,
    petsc_tao,
    ceres
};

// Where the derivatives used by a solve came from. RNC-LM records supplied
// when the caller passed curve derivatives. Other kernels leave the field unset.
enum class derivative_mode : std::uint8_t
{
    supplied
};

}  // namespace solverslib::api

#endif  // SOLVERS_STATUS_H_
