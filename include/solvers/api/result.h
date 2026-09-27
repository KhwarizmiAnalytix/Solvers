#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "include/detail/eigen_support.h"
#include "include/detail/support.h"
#include "solvers/api/status.h"

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

    // 0.5 * ||r(x)||^2 for least squares, or the objective value otherwise.
    double objective = 0.0;

    // Reported quantities at the returned iterate. Optional where an adapter
    // does not expose them.
    std::optional<double> residual_norm;
    std::optional<double> gradient_norm;
    std::optional<double> step_norm;

    std::size_t iterations           = 0;
    std::size_t residual_evaluations = 0;
    std::size_t jacobian_evaluations = 0;
    std::size_t gradient_evaluations = 0;
    std::size_t accepted_steps       = 0;
    std::size_t rejected_steps       = 0;

    // What actually ran, after Auto resolution.
    api::backend   backend   = api::backend::automatic;
    api::algorithm algorithm = api::algorithm::automatic;

    // Preserved backend-specific status code, when one exists.
    std::optional<int> backend_status;

    std::string message;

    bool converged() const noexcept { return status == solver_status::converged; }

    // A limit-hit run still returns a usable iterate; hard failures do not.
    bool has_usable_iterate() const noexcept
    {
        return status == solver_status::converged || status == solver_status::max_iterations;
    }
};
}  // namespace solverslib::api
