#include <cmath>
#include <cstdlib>
#include <limits>

#include "quadrature/gaussian_quadrature.h"
#include "terminals/vector.h"
#include "quarismaTest.h"

namespace
{
void test_hermite()
{
    std::array<size_t, 11> num_of_points = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    double max_r = 0.;
    double max_w = 0.;

    for (const auto k : num_of_points)
    {
        size_t n = 2 * k;

        quarisma::vector<double> roots(n);
        quarisma::vector<double> weights(n);
        quarisma::vector<double> roots_precomputed(n);
        quarisma::vector<double> weights_precomputed(n);

        quarisma::gaussian_quadrature::gauss_hermite_coefficients(n, roots, weights, false);
        quarisma::gaussian_quadrature::gauss_hermite_coefficients(
            n, roots_precomputed, weights_precomputed, true);

        for (size_t i = 0; i < n; ++i)
        {
            max_r = std::fmax(std::abs(roots[i] - roots_precomputed[i]), max_r);
            max_w = std::fmax(std::abs(weights[i] - weights_precomputed[i]), max_w);
        }
    }

    EXPECT_LE(max_w, 10 * std::numeric_limits<double>::epsilon());
    EXPECT_LE(max_r, 10 * std::numeric_limits<double>::epsilon());
}

void test_laguerre()
{
    const short            n = 10;
    quarisma::vector<double> roots1(n);
    quarisma::vector<double> weights1(n);
    quarisma::gaussian_quadrature::gauss_laguerre_coefficients(0.5, n, roots1, weights1);

    quarisma::vector<double> roots2(n);
    quarisma::vector<double> weights2(n);

    double max_error = 0.;
    quarisma::gaussian_quadrature::gauss_laguerre_coefficients(1.5, n, roots2, weights2);
    {
        double sum = 0;
        for (size_t i = 0; i < n; ++i)
            sum += weights1[i] * roots1[i] * roots1[i];

        double sum2 = 0;
        for (size_t i = 0; i < n; ++i)
            sum2 += weights2[i] * roots2[i];

        max_error = std::fmax(std::fabs(sum - sum2), max_error);
    }

    QUARISMA_LOGF(INFO, "gaussian_quadrature::gauss_laguerre %.1e", max_error);
    EXPECT_LE(max_error, 100 * std::numeric_limits<double>::epsilon());
}

void test_legendre()
{
    const short            n = 10;
    quarisma::vector<double> roots1(n);
    quarisma::vector<double> weights1(n);

    double max_error = 0.;
    quarisma::gaussian_quadrature::gauss_legendre_coefficients(n, roots1, weights1);
    {
        double sum = 0;
        for (size_t i = 0; i < n; ++i)
            sum += weights1[i] * roots1[i] * roots1[i];
        max_error = std::fmax(std::fabs(sum - 2. / 3.), max_error);
    }

    QUARISMA_LOGF(INFO, "gaussian_quadrature::gauss_legendre %.1e", max_error);
    EXPECT_LE(max_error, 100 * std::numeric_limits<double>::epsilon());
};

void test_gauss_kronrod(size_t n)
{
    quarisma::vector<double> w1(n + 1);
    quarisma::vector<double> w2(n + 1);
    quarisma::vector<double> x(n + 1);
    quarisma::gaussian_quadrature::gauss_kronrod(n, x, w1, w2);

    {
        auto kernel = [](double x) { return x * x; };

        double f = w1[n] * kernel(x[n]);

        for (size_t i = 0; i < n; i++)
        {
            f += w1[i] * kernel(x[i]);
            f += w1[i] * kernel(-x[i]);
        }

        double max_error = std::fabs(f - 2. / 3.);

        if (n > 1)
        {
            QUARISMA_LOGF(INFO, "gaussian_quadrature::gauss_kronrod %.1e", max_error);
            EXPECT_LE(max_error, 1.E-14);
        }
    }
}
}  // namespace

QUARISMATEST(Math, GaussianQuadrature)
{
    START_LOG_TO_FILE_NAME(GaussianQuadrature);

    test_hermite();

    test_laguerre();

    test_legendre();

    test_gauss_kronrod(1);

    test_gauss_kronrod(5);

    test_gauss_kronrod(6);

    test_gauss_kronrod(64);

    END_LOG_TO_FILE_NAME(GaussianQuadrature);
    END_TEST();
}
