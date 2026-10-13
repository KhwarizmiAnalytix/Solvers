#ifndef SOLVERS_SUPPORT_H_
#define SOLVERS_SUPPORT_H_
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

#include "detail/eigen_support.h"
#include "include/logging.h"

// Direct aliases of the Logging library's own macros: SOLVERS_* is this
// project's naming, LOGGING_* does the actual work (message formatting via
// logging::strings::format's "{}" placeholders, verbosity filtering, and -
// for CHECK/THROW - throwing logging::exception per
// logging::get_exception_mode()). Call sites build any manipulator-formatted
// pieces (setw/scientific/setprecision) into a plain string themselves
// first, since logging::strings::format supports only "{}", not format
// specifiers.
// These take an explicit first parameter (cond/format_str) rather than a bare
// "..." that is passed straight through as LOGGING_*(__VA_ARGS__): under
// MSVC's traditional preprocessor (no /Zc:preprocessor), forwarding one
// macro's whole __VA_ARGS__ as a single argument to another variadic macro
// does not get re-split on commas, so e.g. SOLVERS_CHECK(f1 * f2 <= 0.,
// "f1={} f2={}", f1, f2) collapsed via the C comma operator into a condition
// equal to just the last argument (f2) instead of "f1 * f2 <= 0." -- the
// check silently stopped firing as intended. Naming "cond"/"format_str"
// explicitly forces the preprocessor to split the call site's own arguments
// correctly, and ", ##__VA_ARGS__" re-glues them for LOGGING_*.
#define SOLVERS_CHECK(cond, ...) LOGGING_CHECK(cond, ##__VA_ARGS__)
#define SOLVERS_CHECK_DEBUG(cond, ...) LOGGING_CHECK_DEBUG(cond, ##__VA_ARGS__)
#define SOLVERS_CHECK_FINITE_DEBUG(value) LOGGING_CHECK_DEBUG(std::isfinite(value))
#define SOLVERS_THROW(format_str, ...) LOGGING_THROW(format_str, ##__VA_ARGS__)
#define SOLVERS_NOT_IMPLEMENTED(format_str, ...) \
    LOGGING_NOT_IMPLEMENTED(format_str, ##__VA_ARGS__)
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
// absolute: h = step. relative: h = step * |x|. These are not combined.
enum class finite_difference_scale : int
{
    absolute = 0,
    relative = 1
};

inline double finite_difference_increment(
    finite_difference_scale scale, double step, double coordinate)
{
    if (scale == finite_difference_scale::relative)
    {
        return step * std::abs(coordinate);
    }
    return step;
}

// Common callback types used across all solvers.
using function_type = std::function<void(vector_type const&, vector_type&)>;
using jacobian_type = std::function<void(vector_type const&, matrix_type&)>;

template <typename T>
inline bool is_almost_zero(T value, T tolerance = std::numeric_limits<T>::epsilon())
{
    return std::abs(value) < tolerance;
}
}  // namespace solverslib

#endif  // SOLVERS_SUPPORT_H_
