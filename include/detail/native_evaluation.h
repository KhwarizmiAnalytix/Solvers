#ifndef SOLVERS_NATIVE_EVALUATION_H_
#define SOLVERS_NATIVE_EVALUATION_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "detail/eigen_support.h"
#include "solvers/api/detail/evaluator.h"
#include "solvers/api/detail/evaluators.h"

// Shared plumbing for the native kernels: every kernel evaluates through a
// residual_evaluator. A kernel built from raw callbacks (the legacy
// constructors) wraps them in an evaluator for the duration of one solve.
namespace solverslib
{
using native_residual_function = std::function<void(vector_type const&, vector_type&)>;
using native_jacobian_function = std::function<void(vector_type const&, matrix_type&)>;

struct native_evaluator_binding
{
    std::unique_ptr<api::detail::residual_evaluator> owned;
    api::detail::residual_evaluator*                 evaluator = nullptr;
};

// Use `external` when the caller supplied one; otherwise wrap the callbacks,
// differentiating numerically (relative step `fd_step`) when no Jacobian exists.
inline native_evaluator_binding bind_native_evaluator(api::detail::residual_evaluator* external,
    std::size_t                                                                        n,
    std::size_t                                                                        m,
    const native_residual_function&                                                    residuals,
    const native_jacobian_function&                                                    jacobian,
    double                                                                             fd_step)
{
    native_evaluator_binding binding;
    if (external != nullptr)
    {
        binding.evaluator = external;
        return binding;
    }
    if (jacobian)
    {
        binding.owned = std::make_unique<api::detail::callback_residual_evaluator>(
            n, m, residuals, std::optional<native_jacobian_function>(jacobian));
    }
    else
    {
        binding.owned = std::make_unique<api::detail::finite_difference_residual_evaluator>(
            n, m, residuals, fd_step);
    }
    binding.evaluator = binding.owned.get();
    return binding;
}

inline std::string evaluation_failure_message(const api::detail::residual_evaluator& evaluator,
    api::detail::evaluation_status                                                   status,
    const char*                                                                      where)
{
    if (status == api::detail::evaluation_status::fatal_error)
    {
        return std::string(where) + ": " + evaluator.last_error().value_or("evaluation failed");
    }
    return std::string(where) + ": non-finite or invalid value";
}
}  // namespace solverslib

#endif  // SOLVERS_NATIVE_EVALUATION_H_
