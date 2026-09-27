#pragma once

#include <cstddef>

namespace solverslib
{
enum class native_convergence : int
{
    gradient_converged   = 0,
    parameter_converged  = 1,
    function_converged   = 2,
    not_converged        = 3
};

struct native_result
{
    native_convergence status        = native_convergence::not_converged;
    double             residual_norm = 0.0;
    std::size_t        iterations    = 0;

    bool converged() const noexcept
    {
        return status != native_convergence::not_converged;
    }
};
}  // namespace solverslib
