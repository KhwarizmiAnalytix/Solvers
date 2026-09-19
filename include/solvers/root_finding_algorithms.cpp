#include "solvers/root_finding_algorithms.h"

#include <cmath>
#include <limits>

#include "detail/support.h"

namespace solverslib
{
// ---------------------------------------------------------------------------
// bisection
// ---------------------------------------------------------------------------
bool root_finding_algorithms::bisection(function_type const& func,
    double                                                    x1,
    double                                                    x2,
    double&                                                   root,
    const root_finding_options&                               options)
{
    auto f1 = func(x1) - options.function_offset();
    auto f2 = func(x2) - options.function_offset();
    SOLVERS_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

    if (std::fabs(f1) < options.tolerance_function())
    {
        root = x1;
        return true;
    }
    if (std::fabs(f2) < options.tolerance_function())
    {
        root = x2;
        return true;
    }

    double lo = x1, hi = x2, f_lo = f1;

    for (size_t iter = 0; iter < options.max_iterations(); ++iter)
    {
        const double mid   = 0.5 * (lo + hi);
        const double f_mid = func(mid) - options.function_offset();

        if (std::fabs(f_mid) < options.tolerance_function() ||
            0.5 * std::fabs(hi - lo) < options.tolerance_parameter())
        {
            root = mid;
            return true;
        }

        if (f_lo * f_mid <= 0.)
        {
            hi = mid;
        }
        else
        {
            lo   = mid;
            f_lo = f_mid;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// false_position (regula falsi, with the Illinois anti-stalling modification)
// ---------------------------------------------------------------------------
bool root_finding_algorithms::false_position(function_type const& func,
    double                                                        x1,
    double                                                        x2,
    double&                                                       root,
    const root_finding_options&                                  options)
{
    auto f1 = func(x1) - options.function_offset();
    auto f2 = func(x2) - options.function_offset();
    SOLVERS_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

    if (std::fabs(f1) < options.tolerance_function())
    {
        root = x1;
        return true;
    }
    if (std::fabs(f2) < options.tolerance_function())
    {
        root = x2;
        return true;
    }

    double lo = x1, hi = x2, f_lo = f1, f_hi = f2;
    int    stagnant_side = 0;  // -1: hi retained last, +1: lo retained last

    for (size_t iter = 0; iter < options.max_iterations(); ++iter)
    {
        const double x_new = (f_lo * hi - f_hi * lo) / (f_lo - f_hi);
        const double f_new = func(x_new) - options.function_offset();

        if (std::fabs(f_new) < options.tolerance_function() ||
            std::fabs(hi - lo) < options.tolerance_parameter())
        {
            root = x_new;
            return true;
        }

        if (f_lo * f_new < 0.)
        {
            hi   = x_new;
            f_hi = f_new;
            if (stagnant_side == -1)
            {
                f_lo *= 0.5;
            }
            stagnant_side = -1;
        }
        else
        {
            lo   = x_new;
            f_lo = f_new;
            if (stagnant_side == 1)
            {
                f_hi *= 0.5;
            }
            stagnant_side = 1;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// ridders
// ---------------------------------------------------------------------------
bool root_finding_algorithms::ridders(function_type const& func,
    double                                                 x1,
    double                                                 x2,
    double&                                                root,
    const root_finding_options&                            options)
{
    auto f1 = func(x1) - options.function_offset();
    auto f2 = func(x2) - options.function_offset();
    SOLVERS_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

    if (std::fabs(f1) < options.tolerance_function())
    {
        root = x1;
        return true;
    }
    if (std::fabs(f2) < options.tolerance_function())
    {
        root = x2;
        return true;
    }

    double previous_estimate = std::numeric_limits<double>::quiet_NaN();

    for (size_t iter = 0; iter < options.max_iterations(); ++iter)
    {
        const double mid   = 0.5 * (x1 + x2);
        const double f_mid = func(mid) - options.function_offset();

        if (std::fabs(f_mid) < options.tolerance_function())
        {
            root = mid;
            return true;
        }

        const double s = std::sqrt(f_mid * f_mid - f1 * f2);
        if (s == 0.)
        {
            return false;
        }

        const double dx       = (mid - x1) * f_mid / s;
        const double estimate = (f1 >= f2) ? mid + dx : mid - dx;

        if (!std::isnan(previous_estimate) &&
            std::fabs(estimate - previous_estimate) < options.tolerance_parameter())
        {
            root = estimate;
            return true;
        }
        previous_estimate = estimate;

        const double f_estimate = func(estimate) - options.function_offset();
        if (std::fabs(f_estimate) < options.tolerance_function())
        {
            root = estimate;
            return true;
        }

        if (std::copysign(1.0, f_mid) != std::copysign(1.0, f_estimate))
        {
            x1 = mid;
            f1 = f_mid;
            x2 = estimate;
            f2 = f_estimate;
        }
        else if (std::copysign(1.0, f1) != std::copysign(1.0, f_estimate))
        {
            x2 = estimate;
            f2 = f_estimate;
        }
        else
        {
            x1 = estimate;
            f1 = f_estimate;
        }

        if (std::fabs(x2 - x1) < options.tolerance_parameter())
        {
            root = previous_estimate;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// dekker
// ---------------------------------------------------------------------------
bool root_finding_algorithms::dekker(const function_gradient_type& func,
    double                                                         x1,
    double                                                         x2,
    double&                                                        result,
    const root_finding_options&                                    options)
{
    const double tolerance_function  = options.tolerance_function();
    const double tolerance_parametes = options.tolerance_parameter();
    const size_t max_iterations      = options.max_iterations();

    double df_dx;

    auto f1 = func(x1, df_dx);
    auto f2 = func(x2, df_dx);

    SOLVERS_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

    if (std::abs(f1) < tolerance_function)
    {
        result = x1;
        return true;
    }
    if (std::abs(f2) < tolerance_function)
    {
        result = x2;
        return true;
    }

    auto x_plus  = f1 > 0. ? x1 : x2;
    auto x_minus = f1 > 0. ? x2 : x1;

    auto root = 0.5 * (x1 + x2);
    auto dx   = std::fabs(x2 - x1);
    auto f    = func(root, df_dx);

    for (size_t j = 0; j < max_iterations; j++)
    {
        if ((f - (root - x_plus) * df_dx) * (f - (root - x_minus) * df_dx) > 0. ||
            std::fabs(2. * f) > std::fabs(dx * df_dx))
        {
            dx   = .5 * (x_plus - x_minus);
            root = x_minus + dx;
        }
        else
        {
            dx = f / df_dx;
            root -= dx;
        }

        if (std::fabs(dx) < tolerance_parametes)
        {
            result = root;
            return true;
        }

        f = func(root, df_dx);

        if (std::fabs(f) < tolerance_function)
        {
            result = root;
            return true;
        }

        if (f < 0.)
        {
            x_minus = root;
        }
        else
        {
            x_plus = root;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
// brent
// ---------------------------------------------------------------------------
bool root_finding_algorithms::brent(  // NOLINT
    root_finding_algorithms::function_type const& func,
    double                                        x1,
    double                                        x2,
    double&                                       root,
    const root_finding_options&                   options)
{
    const double tolerance_function  = options.tolerance_function();
    const double tolerance_parametes = options.tolerance_parameter();
    const size_t max_iterations      = options.max_iterations();
    const double f_0                 = options.function_offset();

    auto f1 = func(x1) - f_0;
    auto f2 = func(x2) - f_0;
    SOLVERS_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

    auto   c  = x2;
    double d  = 0.;
    double e  = 0.;
    auto   fc = f2;

    for (size_t iter = 0; iter < max_iterations; iter++)
    {
        if (std::fabs(f2) < tolerance_function)
        {
            root = x2;
            return true;
        }
        if (f2 * fc > 0.)
        {
            c  = x1;
            fc = f1;
            e = d = x2 - x1;
        }
        if (std::fabs(fc) < std::fabs(f2))
        {
            x1 = x2;
            x2 = c;
            c  = x1;
            f1 = f2;
            f2 = fc;
            fc = f1;
        }

        const auto epsilon =
            2. * std::numeric_limits<double>::epsilon() * std::fabs(x2) + 0.5 * tolerance_parametes;

        const double xm = (c - x2);
        if (std::fabs(xm) <= epsilon || is_almost_zero(std::fabs(xm) - epsilon) ||
            std::fabs(f2) < tolerance_function)
        {
            root = x2;
            return true;
        }

        if (std::fabs(e) >= epsilon && std::fabs(f1) > std::fabs(f2))
        {
            const auto s = f2 / f1;
            double     p;
            double     q;
            if (x1 == c)
            {
                p = xm * s;
                q = 1. - s;
            }
            else
            {
                q            = f1 / fc;
                const auto r = s * q;
                p            = s * (xm * q * (q - r) - (x2 - x1) * (r - 1.));
                q            = (q - 1.) * (r - 1.) * (s - 1.);
            }
            if (p > 0.0)
            {
                q = -q;
            }

            p = std::fabs(p);

            e = d = .5 * xm;
            if (2. * p < std::fmin(1.5 * xm * q - std::fabs(epsilon * q), std::fabs(e * q)))
            {
                e = d;
                d = p / q;
            }
        }
        else
        {
            e = d = .5 * xm;
        }

        x1 = x2;
        f1 = f2;
        x2 += (std::fabs(d) > epsilon) ? d : std::copysign(epsilon, xm);

        f2 = func(x2) - f_0;
    }
    return false;
}

// ---------------------------------------------------------------------------
// newton_raphson
// ---------------------------------------------------------------------------
bool root_finding_algorithms::newton_raphson(function_gradient_type const& func,
    double                                                                 x0,
    double&                                                                root,
    const root_finding_options&                                            options)
{
    double x = x0;

    for (size_t iter = 0; iter < options.max_iterations(); ++iter)
    {
        double       df_dx = 0.;
        const double f     = func(x, df_dx) - options.function_offset();

        if (std::fabs(f) < options.tolerance_function())
        {
            root = x;
            return true;
        }

        SOLVERS_CHECK(!is_almost_zero(df_dx), "newton_raphson: derivative vanished at x = ", x);

        const double dx = f / df_dx;
        x -= dx;

        if (std::fabs(dx) < options.tolerance_parameter())
        {
            root = x;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// secant
// ---------------------------------------------------------------------------
bool root_finding_algorithms::secant(function_type const& func,
    double                                                x0,
    double                                                x1,
    double&                                               root,
    const root_finding_options&                           options)
{
    double f0 = func(x0) - options.function_offset();
    double f1 = func(x1) - options.function_offset();

    for (size_t iter = 0; iter < options.max_iterations(); ++iter)
    {
        if (std::fabs(f1) < options.tolerance_function())
        {
            root = x1;
            return true;
        }

        const double denominator = f1 - f0;
        SOLVERS_CHECK(
            !is_almost_zero(denominator), "secant: denominator vanished (f(x1) == f(x0))");

        const double x2 = x1 - f1 * (x1 - x0) / denominator;

        if (std::fabs(x2 - x1) < options.tolerance_parameter())
        {
            root = x2;
            return true;
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = func(x2) - options.function_offset();
    }
    return false;
}

}  // namespace solverslib
