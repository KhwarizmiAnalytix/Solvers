#include "polynomial_solver.h"

#include <array>
#include <complex>

#include "detail/support.h"

namespace quarisma
{
//std::vector<std::complex<double>> solve(double a3, double a2, double a1, double a0)
//{
//    std::vector<std::complex<double>> roots;
//
//    // Convert to depressed quartic y^4 + py^2 + qy + r
//    double p = a2 - 3.0 * a3 * a3 / 8.0;
//    double q = a3 * a3 * a3 / 8.0 - a3 * a2 / 2.0 + a1;
//    double r = -3. * a3 * a3 * a3 * a3 / 256.0 + a3 * a3 * a2 / 16. - a3 * a1 / 4 + a0;
//
//    // Solve resolvent cubic
//    double m =
//        polynomial_solver::third_degree_polynomial_solver(p, 0.25 * p * p - r, -q * q * 0.125);
//
//    //y^2+by+c
//    {
//        const auto b     = sqrt(2 * m);
//        const auto c     = p / 2. + m - q / (2. * b);
//        const auto delta = b * b - 4 * c;
//
//        if (delta > 0.0)
//        {
//            roots.emplace_back(std::complex<double>((-b + std::sqrt(delta)) * 0.5 - a3 * 0.25, 0));
//            roots.emplace_back(std::complex<double>((-b - std::sqrt(delta)) * 0.5 - a3 * 0.25, 0));
//        }
//        else
//        {
//            roots.emplace_back(std::complex<double>(-b - a3 / 4, std::sqrt(-delta)));
//            roots.emplace_back(std::complex<double>(-b - a3 / 4, -std::sqrt(-delta)));
//        }
//    }
//    {
//        const auto b     = -sqrt(2 * m);
//        const auto c     = p / 2. + m - q / (2. * b);
//        const auto delta = b * b - 4 * c;
//
//        if (delta > 0.0)
//        {
//            roots.emplace_back({(-b + std::sqrt(delta)) * 0.5 - a3 * 0.25, 0
//        });
//            roots.emplace_back((-b - std::sqrt(delta)) * 0.5 - a3 * 0.25, 0));
//        }
//        else
//        {
//            roots.emplace_back(std::complex<double>(-b - a3 / 4, std::sqrt(-delta)));
//            roots.emplace_back(std::complex<double>(-b - a3 / 4, -std::sqrt(-delta)));
//        }
//    }
//    return roots;
//}

//------------------------------------------------------------------------------
double polynomial_solver::fourth_degree_polynomial_solver(
    double a3, double a2, double a1, double a0, double threshold)
{
    // Convert to depressed quartic y^4 + py^2 + qy + r
    const auto a3_2 = a3 * a3;
    double     p    = a2 - 3.0 * a3_2 / 8.0;
    double     q    = a3 * a3_2 / 8.0 - a3 * a2 / 2.0 + a1;
    double     r    = -3. * a3_2 * a3_2 / 256.0 + a3_2 * a2 / 16. - a3 * a1 / 4.0 + a0;

    // Solve resolvent cubic
    double m =
        polynomial_solver::third_degree_polynomial_solver(p, 0.25 * p * p - r, -q * q * 0.125);

    double root = std::numeric_limits<double>::min();

    {
        const auto b     = sqrt(2 * m);
        const auto c     = m + p / 2. - q / (2. * b);
        auto       delta = b * b - 4 * c;

        if (delta > 0.0)
        {
            delta           = std::sqrt(delta);
            const auto tmp1 = (-b - delta) * 0.5 - a3 * 0.25;
            const auto tmp2 = (-b + delta) * 0.5 - a3 * 0.25;

            root = std::max({tmp1, tmp2, root});
        }
    }
    {
        const auto b     = -sqrt(2 * m);
        const auto c     = m + p / 2. - q / (2. * b);
        auto       delta = b * b - 4 * c;

        if (delta > 0.0)
        {
            delta           = std::sqrt(delta);
            const auto tmp1 = (-b - delta) * 0.5 - a3 * 0.25;
            const auto tmp2 = (-b + delta) * 0.5 - a3 * 0.25;

            root = std::max({tmp1, tmp2, root});
        }
    }

    SOLVERS_LOG_IF(
        WARNING,
        !is_almost_zero(
            root * root * root * root + a3 * root * root * root + a2 * root * root + a1 * root + a0,
            0.00005),
        "cubic solver has a problem: " << root * root * root * root + a3 * root * root * root +
                                              a2 * root * root + a1 * root + a0);

    return std::max(root, threshold);
}

//------------------------------------------------------------------------------
double polynomial_solver::third_degree_polynomial_solver(double b, double c, double d)
{
    const auto b2 = b * b;
    const auto b3 = b2 * b;

    const auto delta_0    = b2 - 3. * c;
    const auto delta_1    = -b3 + 4.5 * b * c - 13.5 * d;
    const auto delta      = delta_1 * delta_1 - delta_0 * delta_0 * delta_0;
    const auto sqrt_delta = std::sqrt(std::fabs(delta));

    double result = 0.;
    if (delta > 0.)
    {
        const auto term1 = std::cbrt((delta_1 > 0.) ? delta_1 + sqrt_delta : delta_1 - sqrt_delta);
        const auto term2 = delta_0 / term1;

        result = (-b + term1 + term2) * constants::ONE_THIRD;
    }
    else
    {
        const auto norm = sqrt(delta_0);
        const auto phi  = atan2(sqrt_delta, delta_1) * constants::ONE_THIRD;
        result          = (-b + 2. * cos(phi) * norm) * constants::ONE_THIRD;
    }

    SOLVERS_LOG_IF(
        WARNING,
        !is_almost_zero(d + result * (c + result * (b + result)), 0.00001),
        "cubic solver has a problem: " << d + result * (c + result * (b + result)));

    return result;
}
}  // namespace quarisma
