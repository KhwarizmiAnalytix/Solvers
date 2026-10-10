#ifndef SOLVERS_NATIVE_RESULT_H_
#define SOLVERS_NATIVE_RESULT_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace solverslib
{
enum class native_convergence : std::uint8_t
{
    gradient_converged  = 0,
    parameter_converged = 1,
    function_converged  = 2,
    not_converged       = 3,
    numerical_failure = 4,  // an evaluation failed fatally or went non-finite at an accepted point
    stalled           = 5   // no acceptable step could be found; the last accepted point is valid
};

struct native_result
{
    native_convergence status        = native_convergence::not_converged;
    double             residual_norm = 0.0;
    std::size_t        iterations    = 0;
    std::string        message;  // why the run ended; may be empty when it converged

    // Diagnostics at the returned iterate; empty where a kernel does not track them.
    std::optional<double>      gradient_norm;  // ||J^T r|| (or the objective gradient)
    std::optional<double>      step_norm;      // length of the last accepted step
    std::optional<std::size_t> accepted_steps;
    std::optional<std::size_t> rejected_steps;  // trial points evaluated and not accepted

    bool converged() const noexcept
    {
        return status != native_convergence::not_converged &&
               status != native_convergence::numerical_failure &&
               status != native_convergence::stalled;
    }
};
}  // namespace solverslib

#endif  // SOLVERS_NATIVE_RESULT_H_
