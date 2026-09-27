#ifndef SOLVERS_NATIVE_RESULT_H_
#define SOLVERS_NATIVE_RESULT_H_

#include <cstddef>
#include <cstdint>

namespace solverslib
{
enum class native_convergence : std::uint8_t
{
    gradient_converged  = 0,
    parameter_converged = 1,
    function_converged  = 2,
    not_converged       = 3
};

struct native_result
{
    native_convergence status        = native_convergence::not_converged;
    double             residual_norm = 0.0;
    std::size_t        iterations    = 0;

    bool converged() const noexcept { return status != native_convergence::not_converged; }
};
}  // namespace solverslib

#endif  // SOLVERS_NATIVE_RESULT_H_
