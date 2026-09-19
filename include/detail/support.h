#pragma once
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace solverslib::detail
{
template <typename... Args> std::string message(Args&&... args)
{
    std::ostringstream out;
    (void)(out << ... << std::forward<Args>(args));
    return out.str();
}
}  // namespace solverslib::detail
#define SOLVERS_CHECK(condition, ...)                                                              \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
            throw std::invalid_argument(                                                           \
                std::string(#condition) + ": " + ::solverslib::detail::message(__VA_ARGS__));      \
    } while (false)
#define SOLVERS_CHECK_DEBUG(...) SOLVERS_CHECK(__VA_ARGS__)
#define SOLVERS_CHECK_FINITE_DEBUG(value) SOLVERS_CHECK(std::isfinite(value))
#define SOLVERS_THROW(...) throw std::runtime_error(::solverslib::detail::message(__VA_ARGS__))
#define SOLVERS_NOT_IMPLEMENTED(...) SOLVERS_THROW(__VA_ARGS__)
#define SOLVERS_FORCE_INLINE inline
#define SOLVERS_UNUSED [[maybe_unused]]
#define SOLVERS_DELETE_CLASS(T) T() = delete
#define SOLVER_API
#define SOLVER_VISIBILITY
#define SOLVERS_LOGF(level, ...)                                                                   \
    do                                                                                             \
    {                                                                                              \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fputc('\n', stderr);                                                                  \
    } while (false)
#define SOLVERS_LOG_INFO(value)                                                                    \
    do                                                                                             \
    {                                                                                              \
        std::clog << value << '\n';                                                                \
    } while (false)
#define SOLVERS_LOG_ERROR(value) SOLVERS_LOG_INFO(value)
#define SOLVERS_LOG_IF(level, condition, value)                                                    \
    do                                                                                             \
    {                                                                                              \
        if (condition)                                                                             \
        {                                                                                          \
            SOLVERS_LOG_INFO(value);                                                               \
        }                                                                                          \
    } while (false)
namespace solverslib
{
// Canonical dense vector/matrix aliases, shared across every solver_options,
// solver_output, and solver header so they only ever need <Eigen/Core>
// (this header's own include) rather than each pulling in <Eigen/Dense> and
// its decomposition modules.
using vector_type = Eigen::VectorXd;
using matrix_type = Eigen::MatrixXd;

template <typename T>
inline bool is_almost_zero(T value, T tolerance = std::numeric_limits<T>::epsilon())
{
    return std::abs(value) < tolerance;
}
}  // namespace solverslib
