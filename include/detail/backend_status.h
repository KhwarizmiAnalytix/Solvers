#ifndef SOLVERS_BACKEND_STATUS_H_
#define SOLVERS_BACKEND_STATUS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace solverslib
{
// What an external-library adapter (Ipopt, PETSc/TAO) reports about one solve,
// reduced to the distinctions the API cares about. `native_code` keeps the
// library's own code (Ipopt::ApplicationReturnStatus, TaoConvergedReason).
enum class backend_outcome : std::uint8_t
{
    converged,         // a convergence criterion was met
    budget_exhausted,  // iteration/time budget used up; the iterate is usable
    stalled,           // line search / trust region could not make progress
    infeasible,        // the problem was detected to be infeasible
    user_stopped,      // a caller-side callback requested the stop
    failed             // numerical or setup failure
};

struct backend_solve_status
{
    backend_outcome            outcome     = backend_outcome::failed;
    int                        native_code = 0;
    std::optional<std::size_t> iterations;
    std::string                message;

    bool converged() const noexcept { return outcome == backend_outcome::converged; }
};
}  // namespace solverslib

#endif  // SOLVERS_BACKEND_STATUS_H_
