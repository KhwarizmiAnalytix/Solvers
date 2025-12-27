#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include "common/constants.h"
#include "common/macros.h"
#include "distribution/gamma_distribution.h"
#include "distribution/hartman_watson_distribution.h"
#include "distribution/inverse_gaussian_distribution.h"
#include "distribution/normal_distribution.h"
#include "expressions/expressions.h"
#include "laplace_inverter/laplace_inverter_gaver_stehfest.h"
#include "quadrature/gaussian_quadrature.h"
#include "terminals/vector.h"
#include "util/logger.h"
#include "quarismaTest.h"

namespace
{
namespace details
{
void assymptotique_hw(const double t, const quarisma::vector<double>& x, quarisma::vector<double>& h)
{
    const auto cst   = sqrt(t) * quarisma::constants::SQRT_2PI;
    const auto shift = quarisma::constants::PI * quarisma::constants::PI / (2 * t);

    for (size_t i = 0; i < x.size(); i++)
    {
        const auto tmp     = x[i];
        double     rho     = 0.;
        double     drho_dx = 0.;
        if (fabs(tmp) < 0.001)
        {
            auto tmp2 = tmp * tmp;
            rho       = 1. + std::copysign(tmp2 / 6., tmp) + 7. / 360 * tmp2 * tmp2;
            drho_dx =
                1. / 3. * (1. + 7. / 30. * std::copysign(tmp2, tmp) + 31. / 840. * tmp2 * tmp2);
        }
        else if (tmp < 0)
        {
            rho     = tmp / sinh(tmp);
            drho_dx = (tmp / tanh(tmp) - 1.) / (tmp * sinh(tmp));
        }
        else
        {
            rho     = tmp / sin(tmp);
            drho_dx = (1. - tmp / tan(tmp)) / (tmp * sin(tmp));
        }

        h[i] = drho_dx * exp(shift) /** std::cyl_bessel_k(0., rho) */ / (rho * cst);
    }

    h = log(h);
}
}  // namespace details

//tex:
//$$density(u,t) = \theta(u,t)\exp(-u-\frac{t}{8})\frac{\sqrt{2\pi}}{u\sqrt{u}}$$
void test_cheyette_density(
    double                        t,
    size_t                        n,
    const quarisma::vector<double>& hartman_watson_roots,
    const quarisma::vector<double>& hartman_watson_weights)
{
    quarisma::vector<double> weights(2 * n - 1);
    quarisma::vector<double> w2(n);
    quarisma::vector<double> u(n);

    quarisma::gaussian_quadrature::gauss_kronrod(n - 1, u, weights, w2);
    for (size_t i = 0; i < n - 1; ++i)
        weights[2 * n - 2 - i] = weights[i];

    quarisma::vector<double> U(2 * n - 1);
    quarisma::vector<double> density(2 * n - 1);

    quarisma::hartman_watson_distribution::cheyette_density(
        density, U, t, u, weights, hartman_watson_roots, hartman_watson_weights);

    auto result = quarisma::accumulate(density);

    QUARISMA_LOGF(INFO, "time: %f,  %.2e", t, std::fabs(result - 1.));
}

void test_assymptotique()
{
    size_t m = 64;

    quarisma::vector<double> w1(m);
    quarisma::vector<double> w2(m);
    quarisma::vector<double> u(m);

    quarisma::gaussian_quadrature::gauss_kronrod(m - 1, u, w1, w2);

    double t = 20.;

    quarisma::vector<double> r = quarisma::constants::PI * (0.5 - u);

    size_t ng = 26;

    quarisma::vector<double> roots(ng);
    quarisma::vector<double> weights(ng);

    quarisma::gaussian_quadrature::gauss_kronrod(ng - 1, roots, w1, weights);
    {
        quarisma::vector<double> output_asymptotique(m);
        quarisma::hartman_watson_distribution::distribution(
            output_asymptotique,
            t,
            r,
            roots,
            weights,
            quarisma::hartman_watson_distribution_enum::FULLY_ASYMPTOTIC);

        quarisma::vector<double> output_numerical(m);
        quarisma::hartman_watson_distribution::distribution(
            output_asymptotique,
            t,
            r,
            roots,
            weights,
            quarisma::hartman_watson_distribution_enum::NUMERICAL_INTEGRAL);

        quarisma::vector<double> error = output_asymptotique - output_numerical;

        QUARISMA_UNUSED auto max_error = hmax(fabs(error));
    }

    quarisma::vector<double> output_integral(m);
    quarisma::hartman_watson_distribution::distribution(
        output_integral, t, r, roots, weights, quarisma::hartman_watson_distribution_enum::MIXTURE);

    quarisma::vector<double> h_limit(m);
    details::assymptotique_hw(t, r, h_limit);
}

double f(double /*t*/, double /*r*/)
{
    return 1.;
    //std::cyl_bessel_i(sqrt(2. * t), r);
}

double G_asymptotique(double x)
{
    double m;
    double rho;

    if (fabs(x) < 0.01)
    {
        const auto x_2 = x * x;

        rho = 1. + std::copysign(x_2 / 6., x) + 7. / 360 * x_2 * x_2;
        m   = 1 / sqrt(3.) * (1 + x_2 * (std::copysign(1. / 30., x) + 11. / 4200. * x_2));
    }
    else
    {
        if (x < 0.)
        {
            rho = x / sinh(x);

            const auto l           = rho * cosh(x);
            const auto l_minus_one = l - 1.;
            m                      = -sqrt(l_minus_one) / x;
        }
        else
        {
            rho = x / sin(x);

            const auto l          = -rho * cos(x);
            const auto l_plus_one = 1. + l;
            m                     = sqrt(l_plus_one) / x;
        }
    }

    return 1. / m;
}

double theta_asymptotique(double t, double x)
{
    const auto G = G_asymptotique(x);

    const auto r = x == 0. ? 1. / t : (x < 0 ? x / (t * sinh(x)) : x / (t * sin(x)));

    return 2. * G * G *
           exp(quarisma::hartman_watson_distribution::log_distribution_asymptotique(x, t)) /
           (quarisma::constants::SQRT_2PI * exp(-r - 0.125 * t) / (r * sqrt(r)));
}

void test_asymptotique_hartman_watson()
{
    double m = theta_asymptotique(0.1, -5.3697);
    EXPECT_LE(std::fabs(m - 2.0933584667954099e-39), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(0.2, -4.4999);
    EXPECT_LE(std::fabs(m - 1.1736307164661993e-12), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(0.3, -3.9692);
    EXPECT_LE(std::fabs(m - 2.7030914439503544e-06), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(0.5, -3.2638);
    EXPECT_LE(std::fabs(m - 0.011280893212041518), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(1., -2.1773);
    EXPECT_LE(std::fabs(m - 0.26841750049330437), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(1.5, -1.3512);
    EXPECT_LE(std::fabs(m - 0.28965577584675128), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(2.5, quarisma::constants::PI - 2.0105);
    EXPECT_LE(std::fabs(m - 0.16222617584663773), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(3.0, quarisma::constants::PI - 1.6458);
    EXPECT_LE(std::fabs(m - 0.12159788218221429), std::numeric_limits<double>::epsilon());

    m = theta_asymptotique(10., quarisma::constants::PI - 0.5459);
    EXPECT_LE(std::fabs(m - 0.014658905414533291), std::numeric_limits<double>::epsilon());
}

void test_gaver_stehfest()
{
    quarisma::laplace_inverter_gaver_stehfest gaver_stehfest(10, 0.0);

    double t = 0.3;
    double x = -12.;

    // double r = (x == 0. ? 1. / t : (x < 0 ? x / (t * sinh(x)) : x / (t * sin(x))));
    double r = x / (t * sinh(x));

    QUARISMA_UNUSED double tmp = gaver_stehfest(f, t, r);

    QUARISMA_UNUSED double ret = theta_asymptotique(t, x);

    QUARISMA_UNUSED double ret2 = exp(t / 2) * r + exp(2 * t) * r * r;

    QUARISMA_UNUSED double error = tmp - ret;
}

void test_numerical_hartman_watson()
{
    size_t n = 25;

    quarisma::vector<double> roots(n);
    quarisma::vector<double> weights(2 * n - 1);

    quarisma::vector<double> w2(n);
    quarisma::gaussian_quadrature::gauss_kronrod(n - 1, roots, weights, w2);

    for (size_t i = 0; i < n - 1; ++i)
        weights[2 * n - 2 - i] = weights[i];

    auto theta = [&weights, &roots](double t, double x)
    {
        const auto G = G_asymptotique(x);

        const auto r = x == 0. ? 1. / t : (x < 0 ? x / (t * sinh(x)) : x / (t * sin(x)));

        return 2. * G * G *
               quarisma::hartman_watson_distribution::distribution_numerical(x, t, roots, weights) /
               (quarisma::constants::SQRT_2PI * exp(-r - 0.125 * t) / (r * sqrt(r)));
    };

    double m = theta(0.3, -3.9692);
    QUARISMA_LOGF(INFO, "theta value : %.16e", m);
    EXPECT_LE(std::fabs(m - 2.7031575053061699e-06), 5.e-11);

    m = theta(0.5, -3.2638);
    QUARISMA_LOGF(INFO, "theta value : %f", m);
    EXPECT_LE(std::fabs(m - 0.011282022293948941), 1.e-13);

    m = theta(1., -2.1773);
    QUARISMA_LOGF(INFO, "theta value : %f", m);
    EXPECT_LE(std::fabs(m - 0.26855435037598613), 5 * std::numeric_limits<double>::epsilon());

    m = theta(1.5, -1.3512);
    QUARISMA_LOGF(INFO, "theta value : %f", m);
    EXPECT_LE(std::fabs(m - 0.29002119442948870), 5 * std::numeric_limits<double>::epsilon());

    m = theta(2.5, quarisma::constants::PI - 2.0105);
    QUARISMA_LOGF(INFO, "theta value : %f", m);
    EXPECT_LE(std::fabs(m - 0.16281574195428261), 5 * std::numeric_limits<double>::epsilon());

    m = theta(3.0, quarisma::constants::PI - 1.6458);
    QUARISMA_LOGF(INFO, "theta value : %f", m);
    EXPECT_LE(std::fabs(m - 0.12222400966059115), 5 * std::numeric_limits<double>::epsilon());

    m = theta(10., quarisma::constants::PI - 0.5459);
    QUARISMA_LOGF(INFO, "theta value : %f", m);
    EXPECT_LE(std::fabs(m - 0.015120708101036117), 5 * std::numeric_limits<double>::epsilon());
}
}  // namespace

QUARISMATEST(Math, HartmanWatsonDistribution)
{
    test_gaver_stehfest();
    test_asymptotique_hartman_watson();
    test_numerical_hartman_watson();

    size_t ng = 26;

    quarisma::vector<double> roots(ng);
    quarisma::vector<double> weights(ng);

    quarisma::vector<double> w1(ng);

    quarisma::gaussian_quadrature::gauss_kronrod(ng - 1, roots, weights, w1);

    std::vector<double> dates = {
        0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04,
        0.05,       0.051,     0.06,     0.07,    0.075,  0.08,  0.085, 0.09, 0.1,  0.25, 0.45,
        0.5,        0.7,       0.75,     1.,      1.5,    2.,    3.,    4.,   5.,   6.,   7.,
        8.,         9.,        10.,      10.5,    11.,    11.5,  12.,   12.5, 13.,  13.5, 14.,
        14.5,       15.,       15.5,     16.,     16.5,   17.,   17.5,  18.,  18.5, 19.,  19.5,
        20.,        25.,       30.,      35.,     40.,    45.,   50,    100};

    size_t n = 64;
    for (const auto d : dates)
        test_cheyette_density(d, n, roots, weights);

    test_assymptotique();

    END_TEST();
}
