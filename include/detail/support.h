#pragma once
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "include/detail/eigen_support.h"
#include "include/logging.h"

// Direct aliases of the Logging library's own macros: SOLVERS_* is this
// project's naming, LOGGING_* does the actual work (message formatting via
// logging::strings::format's "{}" placeholders, verbosity filtering, and -
// for CHECK/THROW - throwing logging::exception per
// logging::get_exception_mode()). Call sites build any manipulator-formatted
// pieces (setw/scientific/setprecision) into a plain string themselves
// first, since logging::strings::format supports only "{}", not format
// specifiers.
#define SOLVERS_CHECK(...) LOGGING_CHECK(__VA_ARGS__)
#define SOLVERS_CHECK_DEBUG(...) LOGGING_CHECK_DEBUG(__VA_ARGS__)
#define SOLVERS_CHECK_FINITE_DEBUG(value) LOGGING_CHECK_DEBUG(std::isfinite(value))
#define SOLVERS_THROW(...) LOGGING_THROW(__VA_ARGS__)
#define SOLVERS_NOT_IMPLEMENTED(...) LOGGING_NOT_IMPLEMENTED(__VA_ARGS__)
#define SOLVERS_FORCE_INLINE inline
#define SOLVERS_UNUSED [[maybe_unused]]
#define SOLVERS_DELETE_CLASS(T) T() = delete
#define SOLVER_API
#define SOLVER_VISIBILITY

#define SOLVERS_LOGF(level, ...) LOGGING_LOG(level, __VA_ARGS__)
#define SOLVERS_LOG_INFO(...) LOGGING_LOG_INFO(__VA_ARGS__)
#define SOLVERS_LOG_ERROR(...) LOGGING_LOG_ERROR(__VA_ARGS__)
#define SOLVERS_LOG_IF(level, condition, ...) LOGGING_LOG_IF(level, condition, __VA_ARGS__)

namespace solverslib
{
template <typename T>
inline bool is_almost_zero(T value, T tolerance = std::numeric_limits<T>::epsilon())
{
    return std::abs(value) < tolerance;
}
}  // namespace solverslib
