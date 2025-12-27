#include <cmath>
#include <cstddef>

#include "common/discretization.h"
#include "common/finite_difference.h"
#include "grid/space_grid.h"
#include "quarismaTest.h"

using namespace quarisma;

namespace
{
void test_derivatives(bool isuniform)
{
    size_t stencil = 5;

    size_t nx = 201;

    vector<double> x(nx);
    if (isuniform)
    {
        quarisma::discretization::uniform(x.data(), nx, -5., 5., 0.);
    }
    else
    {
        quarisma::discretization::asinh(x.data(), nx, -4., 5., 0.);
    }

    vector<double> f     = x * x * x * x + 0.2 * x * x + 5. * x + 1;
    vector<double> f_d_1 = (4. * x * x * x + 0.4 * x + 5.);
    vector<double> f_d_2 = (12. * x * x + 0.4);

    vector<double> drift(stencil);
    vector<double> vol(stencil);

    size_t i = 0;

    {
        double error_con  = 0.;
        double error_diff = 0.;

        auto* conv = drift.data();
        auto* diff = vol.data();

        finite_difference::nonuniform_five_points_derivative_0(&x[0], i, conv, diff);
        vector<double> V(&f[i - 2], stencil);

        error_con  = std::max(error_con, std::fabs(accumulate(drift * V) - f_d_1[i]));
        error_diff = std::max(error_diff, std::fabs(accumulate(vol * V) - f_d_2[i]));

        QUARISMA_LOG_INFO("convection diff: " << -error_con / f_d_1[i]);
        QUARISMA_LOG_INFO("diffusion diff : " << -error_diff / f_d_2[i]);

        i++;
    }

    {
        double error_con  = 0.;
        double error_diff = 0.;

        auto* conv = drift.data();
        auto* diff = vol.data();

        finite_difference::nonuniform_five_points_derivative_1(&x[0], i, conv, diff);
        vector<double> V(&f[i - 2], stencil);

        error_con  = std::max(error_con, std::fabs(accumulate(drift * V) - f_d_1[i]));
        error_diff = std::max(error_diff, std::fabs(accumulate(vol * V) - f_d_2[i]));

        QUARISMA_LOG_INFO("convection diff: " << -error_con / f_d_1[i]);
        QUARISMA_LOG_INFO("diffusion diff : " << -error_diff / f_d_2[i]);

        i++;
    }

    {
        double error_con  = 0.;
        double error_diff = 0.;

        for (; i < nx - 2; ++i)
        {
            auto* conv = drift.data();
            auto* diff = vol.data();

            finite_difference::nonuniform_five_points_derivative_mid(&x[0], i, conv, diff, false);
            vector<double> V(&f[i - 2], stencil);
            error_con  = std::max(error_con, std::fabs(accumulate(drift * V) / f_d_1[i] - 1.));
            error_diff = std::max(error_diff, std::fabs(accumulate(vol * V) / f_d_2[i] - 1.));
        }

        EXPECT_LE(error_con, 5.E-10);
        EXPECT_LE(error_diff, 5.E-10);
    }

    {
        double error_con  = 0.;
        double error_diff = 0.;

        auto* conv = drift.data();
        auto* diff = vol.data();

        finite_difference::nonuniform_five_points_derivative_1(&x[0], i, conv, diff);
        vector<double> V(&f[i - 2], 5);

        error_con  = std::max(error_con, std::fabs(accumulate(drift * V) - f_d_1[i]));
        error_diff = std::max(error_diff, std::fabs(accumulate(vol * V) - f_d_2[i]));

        QUARISMA_LOG_INFO("convection diff: " << -error_con / f_d_1[i]);
        QUARISMA_LOG_INFO("diffusion diff : " << -error_diff / f_d_2[i]);

        i++;
    }

    {
        double error_con  = 0.;
        double error_diff = 0.;

        auto* conv = drift.data();
        auto* diff = vol.data();

        finite_difference::nonuniform_five_points_derivative_0(&x[0], i, conv, diff);
        vector<double> V(&f[i - 2], 5);

        error_con  = std::max(error_con, std::fabs(accumulate(drift * V) - f_d_1[i]));
        error_diff = std::max(error_diff, std::fabs(accumulate(vol * V) - f_d_2[i]));

        QUARISMA_LOG_INFO("convection diff: " << -error_con / f_d_1[i]);
        QUARISMA_LOG_INFO("diffusion diff : " << -error_diff / f_d_2[i]);

        i++;
    }
}
}  // namespace

QUARISMATEST(Math, FiniteDifference)
{
    START_LOG_TO_FILE_NAME(FiniteDifference);

    test_derivatives(true);
    test_derivatives(false);

    END_LOG_TO_FILE_NAME(FiniteDifference);
    END_TEST();
}
