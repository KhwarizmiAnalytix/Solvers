#ifndef SOLVERS_RESULT_H_
#define SOLVERS_RESULT_H_

#include <cstddef>
#include <optional>
#include <string>

#include "api/status.h"
#include "detail/eigen_support.h"
#include "detail/support.h"

namespace solverslib::api
{
// One structured result for every backend. `converged` is kept separate from
// `has_usable_iterate`: a limit can leave a useful iterate without proving
// convergence. Diagnostics an adapter cannot supply are left as std::nullopt so
// that zero never has to mean "unknown" (review: F08, invariant 1/5).
struct solver_result
{
    solver_status status = solver_status::invalid_problem;

    // Final iterate. Always the last *accepted* point; a rejected trial never
    // replaces it (review invariant 2).
    vector_type parameters;

    // ||r(x)||^2 for least squares (every backend), or the objective value otherwise.
    double objective = 0.0;

    // Reported quantities at the returned iterate. Optional where an adapter
    // does not expose them.
    std::optional<double> residual_norm;
    std::optional<double> gradient_norm;
    std::optional<double> step_norm;

    std::size_t iterations = 0;

    // Work counters. nullopt means the backend does not expose the quantity;
    // an engaged zero means it was measured and nothing happened.
    // residual_evaluations counts executions of the residual function, including
    // those inside finite-difference stencils; objective_evaluations is the
    // scalar-objective counterpart. A rejected step is a trial point that was
    // evaluated and not accepted.
    std::optional<std::size_t> residual_evaluations;
    std::optional<std::size_t> jacobian_evaluations;
    std::optional<std::size_t> objective_evaluations;
    std::optional<std::size_t> gradient_evaluations;
    std::optional<std::size_t> accepted_steps;
    std::optional<std::size_t> rejected_steps;

    // Effective derivative source that was actually used
    std::optional<api::derivative_mode> effective_derivative_source;

    // What actually ran, after Auto resolution.
    api::backend   backend   = api::backend::automatic;
    api::algorithm algorithm = api::algorithm::automatic;

    // Preserved backend-specific status code, when one exists.
    std::optional<int> backend_status;

    std::string message;

    bool converged() const noexcept { return status == solver_status::converged; }

    // A limit-hit or stalled run still returns a usable iterate (the last
    // accepted point); hard failures do not.
    bool has_usable_iterate() const noexcept
    {
        return status == solver_status::converged || status == solver_status::max_iterations ||
               status == solver_status::stalled || status == solver_status::user_stopped;
    }
};
}  // namespace solverslib::api

#endif  // SOLVERS_RESULT_H_
