#include "AADTest.h"
#include "common/macros.h"
#include "common/pointer.h"
#include "interpolator/interpolator_composit.h"
#include "interpolator/interpolator_cubic_hermite.h"
#include "interpolator/interpolator_enum.h"
#include "interpolator/interpolator_factory.h"
#include "interpolator/interpolator_linear.h"
#include "terminals/matrix.h"
#include "quarismaTest.h"

using namespace quarisma;

namespace
{
void testInterpolatorAAD(interpolation_enum enum1, interpolation_enum enum2)
{
    std::vector<double> x{1.0, 1.5, 3.0, 4.1, 7.9};

    std::vector<double> y(x.size());
    for (auto& tmp : y)
    {
        tmp = (double)rand() / RAND_MAX;
    }

    std::vector<double> x_v{1.0, 1.2, 1.5, 2.01, 3.0};
    std::vector<double> w(x_v.size());
    for (auto& tmp : w)
    {
        tmp = (double)rand() / RAND_MAX;
    }

    auto switching_point = 3.0;

    std::vector<double> state_parameters_aad;
    {
        auto cubic = interpolator_dual<std::vector<double>, double, double>(
            x, y, {switching_point}, std::move(std::array<interpolation_enum, 2>{enum1, enum2}));

        state_parameters_aad.resize(cubic.state_parameters_size());

        size_t i = 0;
        for (const auto& tmp : x_v)
        {
            cubic.interpolate_aad(w[i], tmp, state_parameters_aad.data());
            i++;
        }
        cubic.finalize_aad(state_parameters_aad.data());
    }

    auto function = [&x, &x_v, &w, enum1, enum2, switching_point](std::vector<double> y)
    {
        std::vector<double> x_tmp(x.begin(), x.end());
        auto                cubic = interpolator_dual<std::vector<double>, double, double>(
            std::move(x_tmp),
            std::move(y),
            std::move(std::array<double, 1>{switching_point}),
            std::move(std::array<interpolation_enum, 2>{enum1, enum2}));

        size_t i   = 0;
        double ret = 0.;
        for (const auto& tmp : x_v)
        {
            ret += cubic.interpolate(tmp) * w[i];
            i++;
        }

        return ret;
    };

    EXPECT_TRUE(aad_test::run_aad_test(function, y, state_parameters_aad));
}

void testInterpolatorAAD(interpolation_enum type)
{
    std::vector<double> x{1.0, 1.5, 3.0, 4.1, 7.9};

    std::vector<double> y(x.size());
    for (auto& tmp : y)
    {
        tmp = (double)rand() / RAND_MAX;
    }

    std::vector<double> x_v{1.0, 1.5, 2.01, 3.7, 4.6, 4.8, 7.9};
    std::vector<double> w(x_v.size());
    for (auto& tmp : w)
    {
        tmp = (double)rand() / RAND_MAX;
    }

    std::vector<double> state_parameters_aad;
    {
        std::vector<double> x_tmp(x.begin(), x.end());
        std::vector<double> y_tmp(y.begin(), y.end());
        auto cubic = interpolator_factory<std::vector<double>, double, double>::create_ptr(
            type, std::move(x_tmp), std::move(y_tmp));

        state_parameters_aad.resize(cubic->state_parameters_size());

        size_t i = 0;
        for (const auto& tmp : x_v)
        {
            cubic->interpolate_aad(w[i], tmp, state_parameters_aad.data());
            i++;
        }
        cubic->finalize_aad(state_parameters_aad.data());
    }

    auto function = [type, &x, &x_v, &w](std::vector<double> y)
    {
        std::vector<double> x_tmp(x.begin(), x.end());
        auto cubic = interpolator_factory<std::vector<double>, double, double>::create_ptr(
            type, std::move(x_tmp), std::move(y));

        size_t i   = 0;
        double ret = 0.;
        for (const auto& tmp : x_v)
        {
            ret += cubic->interpolate(tmp) * w[i];
            i++;
        }

        return ret;
    };

    EXPECT_TRUE(aad_test::run_aad_test(function, y, state_parameters_aad));
}

void test_cubic_spline(
    std::vector<double>         x,
    std::vector<double>         y,
    double                      alpha,
    cubic_spline_condition_enum left,
    cubic_spline_condition_enum right,
    double                      left_value,
    double                      right_value)
{
    auto cubic =
        util::make_ptr_unique_const<interpolator_cubic_spline<std::vector<double>, double, double>>(
            x, y, left, left_value, right, right_value);

    size_t i = 0;
    for (const auto& tmp : x)
    {
        EXPECT_DOUBLE_EQ(cubic->interpolate(tmp), y[i]);
        i++;
    }

    vector<double> in(1., 7.9, 100);
    vector<double> out(100);
    i = 0;
    for (const auto& tmp : in)
    {
        out[i] = cubic->interpolate(tmp);
        i++;
    }

    if (alpha == 1.)
    {
        out -= (0.5 + (0.1 + (0.02 + 0.005 * in) * in) * in);
        const auto dif = hmax(fabs(out));
        EXPECT_NEAR(dif, 0., 1e-14);
    }

    i = 0;
}
}  // namespace

QUARISMATEST(Math, Interpolation)
{
    START_LOG_TO_FILE_NAME(interpolator);
    // Test Linear Interpolator
    {
        std::vector<double> x{1.0, 1.5, 3.0, 4.1, 7.9};
        std::vector<double> y(x.size());
        for (auto& tmp : y)
        {
            tmp = (double)rand() / RAND_MAX;
        }

        auto linear =
            util::make_ptr_unique_const<interpolator_linear<std::vector<double>, double, double>>(
                x, y);

        size_t i = 0;
        for (const auto& tmp : x)
        {
            EXPECT_DOUBLE_EQ(linear->interpolate(tmp), y[i]);
            i++;
        }

        testInterpolatorAAD(interpolation_enum::LINEAR);
    }

    // Test Cubic Hermite Interpolator
    {
        std::vector<double> x{1.0, 1.5, 3.0, 4.1, 7.9};
        std::vector<double> y(x.size());
        for (auto& tmp : y)
        {
            tmp = rand() / RAND_MAX;
        }

        auto cubic = util::make_ptr_unique_const<
            interpolator_cubic_spline<std::vector<double>, double, double>>(
            std::move(x), std::move(y));

        size_t i = 0;
        for (const auto& tmp : x)
        {
            EXPECT_DOUBLE_EQ(cubic->interpolate(tmp), y[i]);
            i++;
        }

        // Test smoothness
        double h           = 1e-6;
        double x_test      = 2.0;
        double left_deriv  = (cubic->interpolate(x_test) - cubic->interpolate(x_test - h)) / h;
        double right_deriv = (cubic->interpolate(x_test + h) - cubic->interpolate(x_test)) / h;
        EXPECT_NEAR(left_deriv, right_deriv, 5e-6);

        testInterpolatorAAD(interpolation_enum::CUBIC_HERMITE);
    }

    // Test Cubic Spline Interpolator
    {
        std::vector<double> x{1.0, 1.5, 3.0, 4.1, 7.9};
        std::vector<double> y(x.size());
        for (auto& tmp : y)
        {
            tmp = ((double)rand()) / RAND_MAX;
        }

        test_cubic_spline(
            x,
            y,
            0.,
            cubic_spline_condition_enum::FIRST_DERIVATIVE,
            cubic_spline_condition_enum::FIRST_DERIVATIVE,
            0.,
            0.);

        test_cubic_spline(
            x,
            y,
            0.,
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
            0.,
            0.);
        size_t i = 0;
        for (auto& tmp : x)
        {
            y[i] = log(exp(tmp) + 2.);
            i++;
        }

        test_cubic_spline(
            x,
            y,
            0.,
            cubic_spline_condition_enum::FIRST_DERIVATIVE,
            cubic_spline_condition_enum::FIRST_DERIVATIVE,
            (exp(x[0]) + 2.) / exp(x[0]),
            (exp(x.back()) + 2.) / exp(x.back()));

        test_cubic_spline(
            x,
            y,
            0.,
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
            0.,
            0.);

        test_cubic_spline(
            x,
            y,
            0.,
            cubic_spline_condition_enum::NOT_A_KNOT,
            cubic_spline_condition_enum::NOT_A_KNOT,
            0.,
            0.);

        i = 0;
        for (auto& tmp : x)
        {
            y[i] = 0.5 + (0.1 + (0.02 + 0.005 * tmp) * tmp) * tmp;
            i++;
        }

        test_cubic_spline(
            x,
            y,
            1.,
            cubic_spline_condition_enum::FIRST_DERIVATIVE,
            cubic_spline_condition_enum::FIRST_DERIVATIVE,
            0.1 + (0.04 + 3. * 0.005 * x[0]) * x[0],
            0.1 + (0.04 + 3. * 0.005 * x.back()) * x.back());

        test_cubic_spline(
            x,
            y,
            1.,
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
            cubic_spline_condition_enum::SECOND_DERIVATIVE,
            (0.04 + 6. * 0.005 * x[0]),
            (0.04 + 6. * 0.005 * x.back()));

        test_cubic_spline(
            x,
            y,
            0.,
            cubic_spline_condition_enum::NOT_A_KNOT,
            cubic_spline_condition_enum::NOT_A_KNOT,
            0.,
            0.);

        testInterpolatorAAD(interpolation_enum::CUBIC_SPLINE);
    }
    // Test Composite Interpolator - Basic Functionality
    {
        auto linear =
            util::make_ptr_unique_const<interpolator_linear<std::vector<double>, double, double>>(
                std::move(std::vector<double>{1.0, 2.0, 3.0}),
                std::move(std::vector<double>{2.0, 4.0, 8.0}));

        auto geometric = util::make_ptr_unique_const<
            interpolator_cubic_spline<std::vector<double>, double, double>>(
            std::move(std::vector<double>{3.0, 4.0, 5.0}),
            std::move(std::vector<double>{8.0, 16.0, 32.0}));

        std::array<double, 1> switch_points = {3.0};
        interpolator_composite<
            std::vector<double>,
            double,
            double,
            interpolator<std::vector<double>, double, double>,
            interpolator<std::vector<double>, double, double>>
            comp(switch_points, std::move(linear), std::move(geometric));

        auto r = comp.interpolate(1.5);
        EXPECT_DOUBLE_EQ(r, 3.0);
        r = comp.interpolate(3.5);
        EXPECT_NEAR(r, 11.25, 1E-14);
    }

    // Test Composite Interpolator - Initialize Function
    {
        interpolator_dual<std::vector<double>, double, double> comp(
            std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0},
            std::vector<double>{2.0, 4.0, 8.0, 16.0, 32.0},
            std::array<double, 1>{3.0},
            std::array<interpolation_enum, 2>{
                interpolation_enum::LINEAR, interpolation_enum::CUBIC_SPLINE});

        // Test multiple initializations (should be idempotent)
        comp.initialize();
        comp.initialize();

        auto r = comp.interpolate(1.5);
        EXPECT_DOUBLE_EQ(r, 3.0);
        r = comp.interpolate(3.5);
        EXPECT_NEAR(r, 10.928571428571429, 1E-14);
    }

    // Test Composite Interpolator - Edge Cases
    {
        // Test with minimal data points
        interpolator_dual<std::vector<double>, double, double> comp_minimal(
            std::vector<double>{1.0, 2.0},
            std::vector<double>{1.0, 2.0},
            std::array<double, 1>{1.5},
            std::array<interpolation_enum, 2>{
                interpolation_enum::LINEAR, interpolation_enum::LINEAR});

        comp_minimal.initialize();
        auto r = comp_minimal.interpolate(1.5);
        EXPECT_DOUBLE_EQ(r, 1.5);
    }
    // Test Composite Interpolator - Performance and Continuity
    {
        // Test with larger dataset for performance validation
        std::vector<double> x_large, y_large;
        for (int i = 0; i <= 100; ++i)
        {
            x_large.push_back(i * 0.1);
            y_large.push_back(std::sin(i * 0.1) + i * 0.01);
        }

        interpolator_dual<std::vector<double>, double, double> comp_large(
            std::move(x_large),
            std::move(y_large),
            std::array<double, 1>{5.0},
            std::array<interpolation_enum, 2>{
                interpolation_enum::CUBIC_SPLINE, interpolation_enum::LINEAR});

        comp_large.initialize();

        // Test continuity at switch point
        auto left_val   = comp_large.interpolate(4.99);
        auto right_val  = comp_large.interpolate(5.01);
        auto switch_val = comp_large.interpolate(5.0);

        // Values should be reasonably close for continuity
        EXPECT_NEAR(left_val, switch_val, 0.1);
        EXPECT_NEAR(right_val, switch_val, 0.1);
    }

    // Test Composite Interpolator - AAD Integration
    {
        interpolator_dual<std::vector<double>, double, double> comp_aad(
            std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0},
            std::vector<double>{2.0, 4.0, 8.0, 16.0, 32.0},
            std::array<double, 1>{3.0},
            std::array<interpolation_enum, 2>{
                interpolation_enum::LINEAR, interpolation_enum::CUBIC_SPLINE});

        comp_aad.initialize();

        // Test AAD functionality
        std::vector<double> aad_state_parameters(comp_aad.state_parameters_size());
        comp_aad.interpolate_aad(1.0, 1.5, aad_state_parameters.data());
        comp_aad.finalize_aad(aad_state_parameters.data());

        // Verify AAD parameters are properly set
        EXPECT_GT(comp_aad.state_parameters_size(), 0);
    }

    // Test Composite Interpolator - Multiple Interpolation Types
    {
        auto test_enum = [&](interpolation_enum type1, interpolation_enum type2)
        { testInterpolatorAAD(type1, type2); };

        test_enum(interpolation_enum::LINEAR, interpolation_enum::CUBIC_SPLINE);
        test_enum(interpolation_enum::CUBIC_SPLINE, interpolation_enum::LINEAR);
        test_enum(interpolation_enum::CUBIC_HERMITE, interpolation_enum::CUBIC_SPLINE);
        test_enum(interpolation_enum::PIECEWISE_CONSTANT_LEFT, interpolation_enum::LINEAR);
        test_enum(interpolation_enum::PIECEWISE_CONSTANT_RIGHT, interpolation_enum::CUBIC_HERMITE);
    }

    // Test Factory
    {
        std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> y{2.0, 4.0, 8.0, 16.0, 32.0};

        auto linear = interpolator_factory<std::vector<double>, double, double>::create(
            interpolation_enum::LINEAR, x, y);
        EXPECT_DOUBLE_EQ(linear->interpolate(1.5), 3.0);

        auto spline = interpolator_factory<std::vector<double>, double, double>::create(
            interpolation_enum::CUBIC_SPLINE, x, y);
        EXPECT_NEAR(spline->interpolate(1.5), 2.8526785714285712, 1e-10);

        auto cubic = interpolator_factory<std::vector<double>, double, double>::create(
            interpolation_enum::CUBIC_HERMITE, x, y);
        EXPECT_DOUBLE_EQ(cubic->interpolate(1.0), 2.0);
    }

    // Test N-Interpolator Template
    {
        using triple = compose_interpolators<std::vector<double>, double, double, 3>;

        auto linear1 =
            util::make_ptr_unique_const<interpolator_linear<std::vector<double>, double, double>>(
                std::move(std::vector<double>{1.0, 2.0}), std::move(std::vector<double>{2.0, 4.0}));
        auto linear2 =
            util::make_ptr_unique_const<interpolator_linear<std::vector<double>, double, double>>(
                std::move(std::vector<double>{2.0, 3.0}), std::move(std::vector<double>{4.0, 8.0}));
        auto linear3 =
            util::make_ptr_unique_const<interpolator_linear<std::vector<double>, double, double>>(
                std::move(std::vector<double>{3.0, 4.0}),
                std::move(std::vector<double>{8.0, 16.0}));

        std::array<double, 2> switch_points = {2.0, 3.0};
        triple comp(switch_points, std::move(linear1), std::move(linear2), std::move(linear3));

        EXPECT_DOUBLE_EQ(comp.interpolate(1.5), 3.0);
        EXPECT_DOUBLE_EQ(comp.interpolate(2.5), 6.0);
        EXPECT_DOUBLE_EQ(comp.interpolate(3.5), 12.0);
    }

    END_LOG_TO_FILE_NAME(interpolator);
    END_TEST();
}