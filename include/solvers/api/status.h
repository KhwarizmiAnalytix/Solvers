#pragma once

#include "include/detail/support.h"

namespace solverslib::api
{
// One termination vocabulary shared across every backend. Adapters translate
// their native codes (Ceres, PETSc/TAO, Ipopt, or the native
// solver_convergence_enum) into exactly one of these. See the redesign review,
// "Return a structured result", for the rationale on keeping this closed set.
enum class solver_status : int
{
    converged,               // a named stopping criterion was met
    max_iterations,          // iteration/evaluation budget exhausted with a usable iterate
    invalid_problem,         // dimensions, callbacks, or options were rejected before solving
    numerical_failure,       // non-finite quantities or a failed linear solve during the run
    infeasible,              // a supplied point/constraint set is infeasible
    unsupported_capability,  // the chosen backend cannot honor a requested capability
    backend_unavailable,     // the requested backend was not compiled in
    user_stopped             // a caller callback requested a stop
};

// Which algorithm was actually run. Kept independent of the backend that ran
// it (PyTorch/Eigen style: the mathematical method is named separately from the
// storage/execution backend that realizes it).
enum class algorithm : int
{
    automatic = 0,
    levenberg_marquardt,
    gauss_newton,
    bfgs,
    lbfgs,
    pounders,
    interior_point,
    newton_krylov
};

// Which library/executor realized the algorithm.
enum class backend : int
{
    automatic = 0,
    native,
    ipopt,
    petsc_tao,
    pounders,
    ceres
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
    case algorithm::bfgs:
        return "bfgs";
    case algorithm::lbfgs:
        return "lbfgs";
    case algorithm::pounders:
        return "pounders";
    case algorithm::interior_point:
        return "interior_point";
    case algorithm::newton_krylov:
        return "newton_krylov";
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
}  // namespace solverslib::api
