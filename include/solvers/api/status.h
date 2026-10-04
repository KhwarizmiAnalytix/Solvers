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

// Which algorithm was actually run. Kept independent of the backend that ran
// it (PyTorch/Eigen style: the mathematical method is named separately from the
// storage/execution backend that realizes it).
enum class algorithm : std::uint8_t
{
    automatic = 0,
    levenberg_marquardt,
    gauss_newton,
    lbfgs,
    pounders,
    interior_point,
    newton_krylov,
    riemann_normal_coordinate_lm
};

// Which library/executor realized the algorithm.
enum class backend : std::uint8_t
{
    automatic = 0,
    native,
    ipopt,
    petsc_tao,
    pounders,
    ceres
};

// How derivatives are computed or selected.
enum class derivative_mode : std::uint8_t
{
    automatic,  // Prefer supplied Jacobian; fallback to AD; fallback to numeric/derivative-free
    supplied,   // Require an engaged, callable Jacobian; error if missing
    automatic_differentiation,  // Require a compatible AD provider; error if missing or unsupported
    finite_difference           // Use explicit numerical differentiation even if AD/supplied exist
};

inline const char* to_string(solver_status status)
{
    switch (status)
    {
    case solver_status::converged:
        return "converged";
    case solver_status::max_iterations:
        return "max_iterations";
    case solver_status::invalid_problem:
        return "invalid_problem";
    case solver_status::numerical_failure:
        return "numerical_failure";
    case solver_status::infeasible:
        return "infeasible";
    case solver_status::unsupported_capability:
        return "unsupported_capability";
    case solver_status::backend_unavailable:
        return "backend_unavailable";
    case solver_status::user_stopped:
        return "user_stopped";
    case solver_status::stalled:
        return "stalled";
    }
    return "unknown";
}

inline const char* to_string(algorithm value)
{
    switch (value)
    {
    case algorithm::automatic:
        return "automatic";
    case algorithm::levenberg_marquardt:
        return "levenberg_marquardt";
    case algorithm::gauss_newton:
        return "gauss_newton";
    case algorithm::lbfgs:
        return "lbfgs";
    case algorithm::pounders:
        return "pounders";
    case algorithm::interior_point:
        return "interior_point";
    case algorithm::newton_krylov:
        return "newton_krylov";
    case algorithm::riemann_normal_coordinate_lm:
        return "riemann_normal_coordinate_lm";
    }
    return "unknown";
}

inline const char* to_string(backend value)
{
    switch (value)
    {
    case backend::automatic:
        return "automatic";
    case backend::native:
        return "native";
    case backend::ipopt:
        return "ipopt";
    case backend::petsc_tao:
        return "petsc_tao";
    case backend::pounders:
        return "pounders";
    case backend::ceres:
        return "ceres";
    }
    return "unknown";
}

inline const char* to_string(derivative_mode value)
{
    switch (value)
    {
    case derivative_mode::automatic:
        return "automatic";
    case derivative_mode::supplied:
        return "supplied";
    case derivative_mode::automatic_differentiation:
        return "automatic_differentiation";
    case derivative_mode::finite_difference:
        return "finite_difference";
    }
    return "unknown";
}
}  // namespace solverslib::api

#endif  // SOLVERS_STATUS_H_
