#ifndef SOLVERS_NATIVE_EVALUATION_H_
#define SOLVERS_NATIVE_EVALUATION_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "detail/eigen_support.h"

// Shared plumbing for the native kernels.
// Since evaluators have been removed, this file is mostly empty now.
namespace solverslib
{
using native_residual_function = std::function<void(vector_type const&, vector_type&)>;
using native_jacobian_function = std::function<void(vector_type const&, matrix_type&)>;
}  // namespace solverslib

#endif  // SOLVERS_NATIVE_EVALUATION_H_
