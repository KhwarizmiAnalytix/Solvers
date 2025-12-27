#include <cmath>    // for fabs
#include <complex>  // for complex
#include <cstddef>  // for size_t
#include <limits>   // for numer...
#include <vector>   // for alloc...

#include "common/constants.h"  // for SQRT_2PI
#include "common/macros.h"     // for QUARISMA...
#include "distribution/gamma_distribution.h"
#include "distribution/gamma_distribution.h"             // for gamma...
#include "distribution/hartman_watson_distribution.h"    // for hartm...
#include "distribution/inverse_gaussian_distribution.h"  // for inver...
#include "distribution/normal_distribution.h"            // for norma...
#include "expressions/expressions.h"                     // for binar...
#include "quadrature/gaussian_quadrature.h"              // for gauss...
#include "terminals/vector.h"                            // for vector
#include "util/logger.h"                                 // for quarisma...
#include "quarismaTest.h"

namespace
{
namespace details
{
double test_normal_inverse_cumulative(double (*N)(double), double (*inv_N)(double))
{
    const double max          = 1.;
    const double min          = 0.;
    const size_t nu_of_points = 1000000;
    const double dx           = (max - min) / (nu_of_points - 1);

    double p   = min;
    double ret = 1.E-11;
    while (p <= 1.)
    {
        const double q = N(inv_N(p));

        ret = std::max(ret, std::fabs(p - q));

        p += dx;
    }

    return ret;
}
}  // namespace details

void test_gamma_distribution()
{
    quarisma::gamma_distribution::gamma(0.75);
    quarisma::gamma_distribution::gamma(0.25);

    auto a = std::lgammal(-0.6);
    auto b = quarisma::gamma_distribution::lgamma(std::complex<double>(-0.6, 0.));

    EXPECT_LE(std::fabs(b.real() - a), 20. * std::numeric_limits<double>::epsilon());

    a = std::lgammal(11.);
    b = quarisma::gamma_distribution::lgamma(std::complex<double>(11., 0.));
    EXPECT_LE(std::fabs(b.real() - a), 20. * std::numeric_limits<double>::epsilon());
}

void test_normal_distribution()
{
    double             x = 8.;
    QUARISMA_UNUSED auto z_1 =
        quarisma::normal_distribution::cdf_over_density(x) -
        quarisma::normal_distribution::cdf(x) / quarisma::normal_distribution::density(x);

    x = 2.3;
    QUARISMA_UNUSED auto z =
        quarisma::normal_distribution::cdf_over_density(x) -
        quarisma::normal_distribution::cdf(x) / quarisma::normal_distribution::density(x);

    EXPECT_LE(
        details::test_normal_inverse_cumulative(
            quarisma::normal_distribution::cdf, quarisma::normal_distribution::inv_cdf_fast),
        3E-10);

    EXPECT_LE(
        details::test_normal_inverse_cumulative(
            quarisma::normal_distribution::cdf, quarisma::normal_distribution::inv_cdf),
        3E-11);

    std::vector<double> v = {0.1, 0.25, 0.5, 0.75, 0.9};
    quarisma::normal_distribution::inv_cdf(v.size(), v.data(), &v[0]);

    EXPECT_LE(
        std::fabs(8 + quarisma::normal_distribution::inv_cdf(quarisma::normal_distribution::cdf(-8.))),
        4.E-9);
    EXPECT_EQ(quarisma::normal_distribution::cdf(38.), 1.);
}

void test_inverse_gaussian_distribution()
{
    auto x      = 1.5;
    auto lambda = 2.;

    QUARISMA_UNUSED auto p1 = quarisma::inverse_gaussian_distribution::density(x, lambda);
    QUARISMA_UNUSED auto p2 = quarisma::inverse_gaussian_distribution::density(-x, -lambda);

    EXPECT_LE(
        quarisma::inverse_gaussian_distribution::cdf(0., lambda),
        std::numeric_limits<double>::epsilon());

    auto P1 = quarisma::inverse_gaussian_distribution::cdf(x, lambda);
    auto P2 = quarisma::inverse_gaussian_distribution::cdf(-x, -lambda);

    EXPECT_LE(std::fabs(P2 - exp(-2. * lambda) * P1), std::numeric_limits<double>::epsilon());

    P1 = quarisma::inverse_gaussian_distribution::cdf(20., lambda);
    P2 = quarisma::inverse_gaussian_distribution::cdf(-20., -lambda);

    EXPECT_LE(std::fabs(P2 - exp(-2. * lambda) * P1), std::numeric_limits<double>::epsilon());
}
}  // namespace

QUARISMATEST(Math, Distributions)
{
    START_LOG_TO_FILE_NAME(Distributions);

    test_gamma_distribution();

    test_normal_distribution();

    test_inverse_gaussian_distribution();

    END_LOG_TO_FILE_NAME(Distributions);
    END_TEST();
}