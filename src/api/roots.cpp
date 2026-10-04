#include "solvers/api/roots.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

#include "detail/root_finding_core.h"
#include "solver_options/root_finding_options.h"

namespace solverslib::api
{
namespace
{
// Thrown from inside the instrumented function to unwind a core that has no
// error channel of its own; always caught in this file.
struct evaluation_error
{
    std::string message;
};

struct counting_state
{
    std::size_t evaluations = 0;
    bool        has_last    = false;
    double      last_x      = 0.0;
    double      last_value  = 0.0;  // f(last_x) - target
};

std::string describe_point(double x, double value)
{
    return "f(" + std::to_string(x) + ") = " + std::to_string(value);
}

std::optional<std::string> validate(const root_options& options)
{
    if (options.max_iterations == 0)
    {
        return "max_iterations must be positive";
    }
    if (!(options.tolerance_function >= 0.0) || !std::isfinite(options.tolerance_function) ||
        !(options.tolerance_parameter >= 0.0) || !std::isfinite(options.tolerance_parameter))
    {
        return "tolerances must be finite and non-negative";
    }
    if (!std::isfinite(options.target))
    {
        return "target must be finite";
    }
    return std::nullopt;
}

root_finding_options to_core_options(const root_options& options)
{
    return root_finding_options_builder()
        .with_max_iterations(options.max_iterations)
        .with_tolerance_function(options.tolerance_function)
        .with_tolerance_parameter(options.tolerance_parameter)
        .with_function_offset(options.target)
        .build();
}

root_result invalid(std::string message)
{
    root_result result;
    result.status  = solver_status::invalid_problem;
    result.message = std::move(message);
    return result;
}

// Wrap `f` so every call is counted and a non-finite value is reported.
scalar_function instrument(const scalar_function& f, double target, counting_state& state)
{
    return [&f, target, &state](double x)
    {
        ++state.evaluations;
        const double raw = f(x);
        if (!std::isfinite(raw))
        {
            throw evaluation_error{
                "the function returned a non-finite value at " + describe_point(x, raw)};
        }
        state.has_last   = true;
        state.last_x     = x;
        state.last_value = raw - target;
        return raw;
    };
}

scalar_function_gradient instrument(
    const scalar_function_gradient& f, double target, counting_state& state)
{
    return [&f, target, &state](double x, double& derivative)
    {
        ++state.evaluations;
        const double raw = f(x, derivative);
        if (!std::isfinite(raw) || !std::isfinite(derivative))
        {
            throw evaluation_error{
                "the function or its derivative returned a non-finite value at " +
                describe_point(x, raw)};
        }
        state.has_last   = true;
        state.last_x     = x;
        state.last_value = raw - target;
        return raw;
    };
}

// Turn a finished (or failed) core run into the structured result, evaluating
// the residual at the returned point only when it is not already known.
template <class Call, class Evaluate>
root_result run_core(counting_state& state, Call&& call, Evaluate&& evaluate_residual)
{
    root_result result;
    try
    {
        const detail::root_run run = call();
        result.root                = run.root;
        result.iterations          = run.iterations;
        switch (run.outcome)
        {
        case detail::root_outcome::converged:
            result.status  = solver_status::converged;
            result.message = "converged";
            break;
        case detail::root_outcome::iteration_limit:
            result.status  = solver_status::max_iterations;
            result.message = "iteration limit reached; root is the best estimate";
            break;
        case detail::root_outcome::degenerate:
            result.status  = solver_status::stalled;
            result.message = "the method could not continue (degenerate interpolation step)";
            break;
        }
        if (state.has_last && state.last_x == run.root)
        {
            result.residual = state.last_value;
        }
        else
        {
            result.residual = evaluate_residual(run.root);
        }
    }
    catch (const evaluation_error& e)
    {
        result.status  = solver_status::numerical_failure;
        result.message = e.message;
        result.root    = std::numeric_limits<double>::quiet_NaN();
    }
    catch (const std::exception& e)
    {
        // Cores report a vanished derivative / interpolation denominator, or a
        // throwing user function, by exception.
        result.status  = solver_status::numerical_failure;
        result.message = e.what();
        result.root    = std::numeric_limits<double>::quiet_NaN();
    }
    result.evaluations = state.evaluations;
    return result;
}
}  // namespace

root_result find_root(
    root_method method, const scalar_function& f, double a, double b, const root_options& options)
{
    if (const auto problem = validate(options))
    {
        return invalid(*problem);
    }
    if (!f || !std::isfinite(a) || !std::isfinite(b))
    {
        return invalid("a function and finite interval endpoints or starting points are required");
    }
    if (method == root_method::dekker || method == root_method::newton_raphson)
    {
        return invalid(std::string(to_string(method)) +
                       " needs a derivative: pass a function returning value and derivative");
    }

    counting_state state;
    const auto     core_f       = instrument(f, options.target, state);
    const auto     core_options = to_core_options(options);
    const auto     residual_at  = [&](double x) { return core_f(x) - options.target; };

    if (method != root_method::secant)
    {
        // Check the bracket here so an unusable interval is reported as a result
        // rather than thrown by the method.
        double fa = 0.0;
        double fb = 0.0;
        try
        {
            fa = core_f(a) - options.target;
            fb = core_f(b) - options.target;
        }
        catch (const evaluation_error& e)
        {
            root_result result;
            result.status      = solver_status::numerical_failure;
            result.message     = e.message;
            result.evaluations = state.evaluations;
            return result;
        }
        catch (const std::exception& e)
        {
            root_result result;
            result.status      = solver_status::numerical_failure;
            result.message     = e.what();
            result.evaluations = state.evaluations;
            return result;
        }
        if (fa * fb > 0.0)
        {
            root_result result = invalid(
                "the interval does not bracket a root: f(a) - target = " + std::to_string(fa) +
                ", f(b) - target = " + std::to_string(fb) + " have the same sign");
            result.evaluations = state.evaluations;
            return result;
        }
    }

    return run_core(
        state,
        [&]() -> detail::root_run
        {
            switch (method)
            {
            case root_method::bisection:
                return detail::run_bisection(core_f, a, b, core_options);
            case root_method::false_position:
                return detail::run_false_position(core_f, a, b, core_options);
            case root_method::ridders:
                return detail::run_ridders(core_f, a, b, core_options);
            case root_method::brent:
                return detail::run_brent(core_f, a, b, core_options);
            case root_method::secant:
                return detail::run_secant(core_f, a, b, core_options);
            default:
                break;
            }
            throw std::logic_error("unreachable root method");
        },
        residual_at);
}

root_result find_root(root_method   method,
    const scalar_function_gradient& f,
    double                          a,
    double                          b,
    const root_options&             options)
{
    if (const auto problem = validate(options))
    {
        return invalid(*problem);
    }
    if (!f || !std::isfinite(a) || !std::isfinite(b))
    {
        return invalid("a function and finite interval endpoints are required");
    }
    if (method != root_method::dekker)
    {
        return invalid(std::string(to_string(method)) +
                       " does not take a bracket with a derivative; use dekker, or the "
                       "value-only or starting-point overloads");
    }

    counting_state state;
    const auto     core_f       = instrument(f, options.target, state);
    const auto     core_options = to_core_options(options);

    double fa = 0.0;
    double fb = 0.0;
    try
    {
        double derivative = 0.0;
        fa                = core_f(a, derivative) - options.target;
        fb                = core_f(b, derivative) - options.target;
    }
    catch (const evaluation_error& e)
    {
        root_result result;
        result.status      = solver_status::numerical_failure;
        result.message     = e.message;
        result.evaluations = state.evaluations;
        return result;
    }
    catch (const std::exception& e)
    {
        root_result result;
        result.status      = solver_status::numerical_failure;
        result.message     = e.what();
        result.evaluations = state.evaluations;
        return result;
    }
    if (fa * fb > 0.0)
    {
        root_result result =
            invalid("the interval does not bracket a root: f(a) - target = " + std::to_string(fa) +
                    ", f(b) - target = " + std::to_string(fb) + " have the same sign");
        result.evaluations = state.evaluations;
        return result;
    }

    return run_core(
        state,
        [&]() { return detail::run_dekker(core_f, a, b, core_options); },
        [&](double x)
        {
            double derivative = 0.0;
            return core_f(x, derivative) - options.target;
        });
}

root_result find_root(
    root_method method, const scalar_function_gradient& f, double x0, const root_options& options)
{
    if (const auto problem = validate(options))
    {
        return invalid(*problem);
    }
    if (!f || !std::isfinite(x0))
    {
        return invalid("a function and a finite starting point are required");
    }
    if (method != root_method::newton_raphson)
    {
        return invalid(std::string(to_string(method)) +
                       " does not take a single starting point with a derivative; use "
                       "newton_raphson");
    }

    counting_state state;
    const auto     core_f       = instrument(f, options.target, state);
    const auto     core_options = to_core_options(options);
    return run_core(
        state,
        [&]() { return detail::run_newton_raphson(core_f, x0, core_options); },
        [&](double x)
        {
            double derivative = 0.0;
            return core_f(x, derivative) - options.target;
        });
}

// -- real polynomial roots ---------------------------------------------------------------
namespace
{
double horner(const std::vector<double>& a, double x)
{
    double value = 0.0;
    for (const double c : a)
    {
        value = value * x + c;
    }
    return value;
}

// Magnitude scale of the Horner evaluation, for a rounding-error bound.
double horner_scale(const std::vector<double>& a, double x)
{
    double value = 0.0;
    for (const double c : a)
    {
        value = value * std::abs(x) + std::abs(c);
    }
    return value;
}

std::vector<double> quadratic_roots(double a, double b, double c)
{
    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0)
    {
        return {};
    }
    if (discriminant == 0.0)
    {
        return {-b / (2.0 * a)};
    }
    // Avoid cancellation: compute the larger-magnitude root first, derive the other.
    const double q  = -0.5 * (b + std::copysign(std::sqrt(discriminant), b));
    double       r1 = q / a;
    double       r2 = c / q;
    if (r1 > r2)
    {
        std::swap(r1, r2);
    }
    return {r1, r2};
}

// Real roots of a polynomial with a[0] != 0, ascending and distinct. Degree 3 and
// above: the real roots of the derivative are the critical points, so p changes
// sign at most once between consecutive ones and bisection finds each root. A
// root at a critical point (a multiple root) is detected by a rounding bound.
std::vector<double> roots_of(const std::vector<double>& a)
{
    const std::size_t degree = a.size() - 1;
    if (degree == 0)
    {
        return {};
    }
    if (degree == 1)
    {
        return {-a[1] / a[0]};
    }
    if (degree == 2)
    {
        return quadratic_roots(a[0], a[1], a[2]);
    }

    std::vector<double> derivative(degree);
    for (std::size_t i = 0; i < degree; ++i)
    {
        derivative[i] = a[i] * static_cast<double>(degree - i);
    }
    const std::vector<double> critical = roots_of(derivative);

    // Cauchy bound: every root lies strictly inside (-R, R).
    double largest = 0.0;
    for (std::size_t i = 1; i < a.size(); ++i)
    {
        largest = std::max(largest, std::abs(a[i] / a[0]));
    }
    const double radius = 1.0 + largest;

    std::vector<double> points;
    points.reserve(critical.size() + 2);
    points.push_back(-radius);
    points.insert(points.end(), critical.begin(), critical.end());
    points.push_back(radius);

    constexpr double    eps = std::numeric_limits<double>::epsilon();
    std::vector<double> values(points.size());
    std::vector<bool>   on_root(points.size(), false);
    std::vector<double> roots;
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        values[i] = horner(a, points[i]);
        // Only interior critical points can be multiple roots; the Cauchy bound
        // endpoints never are.
        if (i > 0 && i + 1 < points.size() &&
            std::abs(values[i]) <=
                8.0 * eps * horner_scale(a, points[i]) * static_cast<double>(degree))
        {
            on_root[i] = true;
            roots.push_back(points[i]);
        }
    }

    for (std::size_t i = 0; i + 1 < points.size(); ++i)
    {
        if (on_root[i] || on_root[i + 1] || !(values[i] * values[i + 1] < 0.0))
        {
            continue;
        }
        double lo = points[i], hi = points[i + 1], f_lo = values[i];
        for (int step = 0; step < 2200; ++step)  // enough to exhaust double precision
        {
            const double mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi)
            {
                break;
            }
            const double f_mid = horner(a, mid);
            if (f_mid == 0.0)
            {
                lo = hi = mid;
                break;
            }
            if ((f_mid < 0.0) == (f_lo < 0.0))
            {
                lo   = mid;
                f_lo = f_mid;
            }
            else
            {
                hi = mid;
            }
        }
        roots.push_back(0.5 * (lo + hi));
    }

    std::sort(roots.begin(), roots.end());
    // Distinct roots only; a multiple root is reported once.
    roots.erase(
        std::unique(roots.begin(),
            roots.end(),
            [](double x, double y)
            { return std::abs(x - y) <= 1e-12 * (1.0 + std::max(std::abs(x), std::abs(y))); }),
        roots.end());
    return roots;
}
}  // namespace

real_roots_result real_roots(const std::vector<double>& coefficients)
{
    real_roots_result result;
    if (coefficients.size() < 2)
    {
        result.message =
            "a polynomial of degree at least one is required (two or more coefficients)";
        return result;
    }
    for (const double c : coefficients)
    {
        if (!std::isfinite(c))
        {
            result.message = "coefficients must be finite";
            return result;
        }
    }
    if (coefficients.front() == 0.0)
    {
        result.message = "the leading coefficient must be nonzero";
        return result;
    }
    result.roots   = roots_of(coefficients);
    result.status  = solver_status::converged;
    result.message = result.roots.empty() ? "no real roots" : "ok";
    return result;
}

real_roots_result real_roots_quadratic(double b, double c)
{
    return real_roots({1.0, b, c});
}

real_roots_result real_roots_cubic(double b, double c, double d)
{
    return real_roots({1.0, b, c, d});
}

real_roots_result real_roots_quartic(double a3, double a2, double a1, double a0)
{
    return real_roots({1.0, a3, a2, a1, a0});
}

std::optional<double> smallest_positive_root(const std::vector<double>& roots)
{
    std::optional<double> best;
    for (const double r : roots)
    {
        if (r > 0.0 && (!best || r < *best))
        {
            best = r;
        }
    }
    return best;
}

std::optional<double> largest_root(const std::vector<double>& roots)
{
    if (roots.empty())
    {
        return std::nullopt;
    }
    return *std::max_element(roots.begin(), roots.end());
}

std::vector<double> roots_in_interval(const std::vector<double>& roots, double lo, double hi)
{
    std::vector<double> selected;
    for (const double r : roots)
    {
        if (r >= lo && r <= hi)
        {
            selected.push_back(r);
        }
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

std::optional<double> smallest_root_in_interval(
    const std::vector<double>& roots, double lo, double hi)
{
    const auto selected = roots_in_interval(roots, lo, hi);
    if (selected.empty())
    {
        return std::nullopt;
    }
    return selected.front();
}

std::optional<double> at_least(const std::optional<double>& root, double floor)
{
    if (!root)
    {
        return std::nullopt;
    }
    return std::max(*root, floor);
}
}  // namespace solverslib::api
