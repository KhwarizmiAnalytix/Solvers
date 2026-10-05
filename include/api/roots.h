#ifndef SOLVERS_API_ROOTS_H_
#define SOLVERS_API_ROOTS_H_

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "detail/support.h"
#include "api/status.h"

// Structured results for scalar root finding and real polynomial roots. They
// extend the problem-structure API's conventions (one closed status vocabulary,
// no exceptions across the boundary, nothing silently clamped) to the 1-D
// solvers. The bool/out-parameter forms in root_finding_algorithms and the
// single-double forms in polynomial_solver remain as convenience wrappers.
namespace solverslib::api
{
enum class root_method : std::uint8_t
{
    bisection,       // bracket, value only
    false_position,  // bracket, value only (Illinois variant)
    ridders,         // bracket, value only
    brent,           // bracket, value only; the robust default
    dekker,          // bracket, needs the derivative
    secant,          // two starting points, value only
    newton_raphson   // one starting point, needs the derivative
};

inline const char* to_string(root_method method)
{
    switch (method)
    {
    case root_method::bisection:
        return "bisection";
    case root_method::false_position:
        return "false_position";
    case root_method::ridders:
        return "ridders";
    case root_method::brent:
        return "brent";
    case root_method::dekker:
        return "dekker";
    case root_method::secant:
        return "secant";
    case root_method::newton_raphson:
        return "newton_raphson";
    }
    return "unknown";
}

// Solve f(x) = target.
struct root_options
{
    std::size_t max_iterations      = 50;
    double      tolerance_function  = std::numeric_limits<double>::epsilon();
    double      tolerance_parameter = std::numeric_limits<double>::epsilon();
    double      target              = 0.0;
};

struct root_result
{
    solver_status status = solver_status::invalid_problem;

    // The root, or the best estimate when the budget ran out (status
    // max_iterations / stalled). NaN when no estimate exists.
    double root = std::numeric_limits<double>::quiet_NaN();
    // f(root) - target at the returned point.
    double residual = std::numeric_limits<double>::quiet_NaN();

    std::size_t iterations = 0;
    // Every call of the function, including bracket checks and the final
    // residual evaluation; a derivative callback counts as one evaluation.
    std::size_t evaluations = 0;

    std::string message;

    bool converged() const noexcept { return status == solver_status::converged; }
    bool has_usable_iterate() const noexcept
    {
        return status == solver_status::converged || status == solver_status::max_iterations ||
               status == solver_status::stalled;
    }
};

using scalar_function          = std::function<double(double)>;
using scalar_function_gradient = std::function<double(double, double&)>;

// Bracketing methods (bisection, false_position, ridders, brent) and secant.
// For the bracketing methods [a, b] must contain a sign change of f - target,
// otherwise the result is invalid_problem and nothing is searched; for secant
// a and b are the two starting points.
root_result find_root(
    root_method method, const scalar_function& f, double a, double b, const root_options& = {});

// dekker: bracket [a, b] with f(x, df_dx) returning the value and the derivative.
root_result find_root(root_method   method,
    const scalar_function_gradient& f,
    double                          a,
    double                          b,
    const root_options& = {});

// newton_raphson from a single starting point.
root_result find_root(
    root_method method, const scalar_function_gradient& f, double x0, const root_options& = {});

// -- real polynomial roots ---------------------------------------------------------------
struct real_roots_result
{
    solver_status       status = solver_status::invalid_problem;
    std::vector<double> roots;  // distinct real roots, ascending; a multiple root appears once
    std::string         message;

    bool converged() const noexcept { return status == solver_status::converged; }
};

// All real roots of a_n x^n + ... + a_0, coefficients highest degree first. The
// leading coefficient must be nonzero and every coefficient finite, otherwise
// invalid_problem. An empty `roots` with status converged means "no real roots".
real_roots_result real_roots(const std::vector<double>& coefficients_highest_first);

// Monic forms matching polynomial_solver:  x^2 + b x + c,  x^3 + b x^2 + c x + d,
// x^4 + a3 x^3 + a2 x^2 + a1 x + a0.
real_roots_result real_roots_quadratic(double b, double c);
real_roots_result real_roots_cubic(double b, double c, double d);
real_roots_result real_roots_quartic(double a3, double a2, double a1, double a0);

// -- selection policies (separate and explicit; nothing is clamped implicitly) ----------
std::optional<double> smallest_positive_root(const std::vector<double>& roots);
std::optional<double> largest_root(const std::vector<double>& roots);
// Roots in the closed interval [lo, hi], ascending.
std::vector<double>   roots_in_interval(const std::vector<double>& roots, double lo, double hi);
std::optional<double> smallest_root_in_interval(
    const std::vector<double>& roots, double lo, double hi);
// The legacy "clamp to threshold" behavior, as an explicit opt-in: max(root, floor),
// or std::nullopt when there is no root.
std::optional<double> at_least(const std::optional<double>& root, double floor);
}  // namespace solverslib::api

#endif  // SOLVERS_API_ROOTS_H_
