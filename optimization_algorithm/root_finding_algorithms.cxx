#include "optimization_algorithm/root_finding_algorithms.h"

#include <cmath>
#include <limits>

#include "common/constants.h"
#include "util/exception.h"

namespace quarisma
{
bool root_finding_algorithms::dekker(
    const function_gradient_type& func,
    double                        x1,
    double                        x2,
    double&                       result,
    double                        tolerance_function,
    double                        tolerance_parametes,
    size_t                        max_iterations)
{
    double df_dx;

    auto f1 = func(x1, df_dx);
    auto f2 = func(x2, df_dx);

    QUARISMA_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

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

bool root_finding_algorithms::brent(  //NOLINT
    root_finding_algorithms::function_type const& func,
    double                                        x1,
    double                                        x2,
    double&                                       root,
    double                                        f_0,
    double                                        tolerance_function,
    double                                        tolerance_parametes,
    size_t                                        max_iterations)
{
    auto f1 = func(x1) - f_0;
    auto f2 = func(x2) - f_0;
    QUARISMA_CHECK(f1 * f2 <= 0., " f1 ", f1, " f2 ", f2);

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
}  // namespace quarisma
