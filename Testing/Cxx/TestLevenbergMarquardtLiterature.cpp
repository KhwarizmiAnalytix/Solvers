#include "solver_options/solver_options_lm.h"
#include "solvers/levenberg_marquardt_solver.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace solverslib;

TEST(LMLiterature, GeodesicAccelerationHasCorrectSignAndRatio)
{
    vector_type x(1);
    x << 2.;
    levenberg_marquardt_solver solver(
        1,
        1,
        [](const vector_type& p, vector_type& r) { r[0] = p[0] * p[0] - 1.; },
        [](const vector_type& p, matrix_type& j) { j(0, 0) = 2. * p[0]; });
    auto         options = solver_options_lm_builder()
                               .with_max_iterations(1)
                               .with_geodesic_acceleration_threshold(.5)
                               .build();
    const double v       = -12. / (16. * 1.0001);
    const double a       = -4. * (2. * v * v) / (16. * 1.0001);
    solver.solve(x, *options);
    EXPECT_NEAR(x[0], 2. + v + .5 * a, 1.e-12);
}

TEST(LMLiterature, ExcessiveAccelerationRejectsEvenDownhillTrial)
{
    vector_type x(1);
    x << 2.;
    levenberg_marquardt_solver solver(
        1,
        1,
        [](const vector_type& p, vector_type& r) { r[0] = p[0] * p[0] - 1.; },
        [](const vector_type& p, matrix_type& j) { j(0, 0) = 2. * p[0]; });
    auto options = solver_options_lm_builder()
                       .with_max_iterations(1)
                       .with_bold_acceptance()
                       .with_geodesic_acceleration_threshold(.1)
                       .build();
    auto result  = solver.solve(x, *options);
    EXPECT_EQ(x[0], 2.);
    EXPECT_FALSE(result.converged());
}

TEST(LMLiterature, BoldAcceptanceAllowsAlignedUphillTrial)
{
    vector_type x(1);
    x << 2.;
    int                        jacobians = 0;
    levenberg_marquardt_solver solver(
        1,
        1,
        [&jacobians](const vector_type&, vector_type& r)
        { r[0] = jacobians == 0 ? 1. : (jacobians == 1 ? .5 : .6); },
        [&jacobians](const vector_type&, matrix_type& j)
        {
            j(0, 0) = 1.;
            ++jacobians;
        });
    auto options = solver_options_lm_builder()
                       .with_max_iterations(2)
                       .with_geodesic_acceleration(false)
                       .with_bold_acceptance()
                       .build();
    auto result  = solver.solve(x, *options);
    EXPECT_DOUBLE_EQ(result.residual_norm, .6);
    EXPECT_EQ(jacobians, 3);
    EXPECT_LT(x[0], 1.);
}

TEST(LMLiterature, StationaryNonzeroResidualConvergesBeforeTrial)
{
    vector_type x(1);
    x << 0.;
    levenberg_marquardt_solver solver(
        1,
        1,
        [](const vector_type&, vector_type& r) { r[0] = 1.; },
        [](const vector_type&, matrix_type& j) { j.setZero(); });
    auto result = solver.solve(x, *solver_options_lm_builder().build());
    EXPECT_EQ(result.status, native_convergence::gradient_converged);
    EXPECT_EQ(result.iterations, 0u);
}

TEST(LMLiterature, MarquardtScalingDampsUnobservableParameter)
{
    vector_type x(2);
    x << 2., 7.;
    levenberg_marquardt_solver solver(
        2,
        1,
        [](const vector_type& p, vector_type& r) { r[0] = p[0] - 1.; },
        [](const vector_type&, matrix_type& j) { j << 1., 0.; });
    auto options = solver_options_lm_builder()
                       .with_type(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT)
                       .with_max_iterations(1)
                       .build();
    solver.solve(x, *options);
    EXPECT_NEAR(x[0], 1.000099990001, 1.e-12);
    EXPECT_EQ(x[1], 7.);
}

TEST(LMLiterature, OptionsAreValidatedAndBuiltSnapshotsAreIndependent)
{
    EXPECT_THROW(solver_options_lm_builder().with_geodesic_acceleration_step(0.).build(),
        std::invalid_argument);
    EXPECT_THROW(solver_options_lm_builder().with_bold_acceptance_exponent(-1.).build(),
        std::invalid_argument);
    EXPECT_THROW(solver_options_lm_builder().with_damping_floor(0.).build(), std::invalid_argument);
    EXPECT_THROW(
        solver_options_lm_builder().with_damping_ceiling(1e-7).build(), std::invalid_argument);
    EXPECT_THROW(
        solver_options_lm_builder().with_roundoff_noise_factor(-1.).build(), std::invalid_argument);
    EXPECT_THROW(solver_options_lm_builder()
                     .with_initial_damping(std::numeric_limits<double>::quiet_NaN())
                     .build(),
        std::invalid_argument);
    const auto configured = solver_options_lm_builder()
                                .with_damping_floor(1e-6)
                                .with_nielsen_damping_floor(1e-14)
                                .with_damping_ceiling(1e10)
                                .with_levenberg_marquardt_damping_ceiling(1e6)
                                .with_diagonal_scaling_floor(1e-10)
                                .with_roundoff_noise_factor(4.)
                                .build();
    EXPECT_DOUBLE_EQ(configured->damping_floor(), 1e-6);
    EXPECT_DOUBLE_EQ(configured->nielsen_damping_floor(), 1e-14);
    EXPECT_DOUBLE_EQ(configured->damping_ceiling(), 1e10);
    EXPECT_DOUBLE_EQ(configured->levenberg_marquardt_damping_ceiling(), 1e6);
    EXPECT_DOUBLE_EQ(configured->diagonal_scaling_floor(), 1e-10);
    EXPECT_DOUBLE_EQ(configured->roundoff_noise_factor(), 4.);
    solver_options_lm_builder builder;
    auto                      first  = builder.with_bold_acceptance().build();
    auto                      second = builder.with_bold_acceptance(false).build();
    EXPECT_TRUE(first->bold_acceptance());
    EXPECT_FALSE(second->bold_acceptance());
    EXPECT_TRUE(first->bold_acceptance());
    EXPECT_TRUE(first->geodesic_acceleration());
}

TEST(LMLiterature, BoldAcceptanceUsesVelocityRatherThanAcceleratedDisplacement)
{
    vector_type                x           = vector_type::Zero(2);
    int                        evaluations = 0;
    int                        jacobians   = 0;
    double                     lambda      = 1.e-4;
    levenberg_marquardt_solver solver(
        2,
        2,
        [&](const vector_type&, vector_type& r)
        {
            const int call = evaluations++;
            r.setZero();
            if (call == 0)
                r[0] = 1.;
            else if (call == 2)
                r[0] = .5;
            else if (call == 4)
                r[0] = 3.;
            else
            {
                const double base = jacobians == 1 ? 1. : .5;
                const double v    = base / (1. + lambda);
                r[0]              = base - .05 * v;
                r[1]              = .5 * .05 * .05 * (jacobians == 1 ? .7 : -.7) * base;
            }
        },
        [&](const vector_type&, matrix_type& j)
        {
            j.setIdentity();
            ++jacobians;
            if (jacobians == 2)
                lambda /= 9.;
        });
    auto       options = solver_options_lm_builder()
                             .with_max_iterations(2)
                             .with_type(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT)
                             .with_bold_acceptance()
                             .build();
    const auto result  = solver.solve(x, *options);
    // Raw velocities are collinear; accelerated displacements would fail
    // Eq. (22): (1 - 0.782)^2 * 9 > 0.25.
    EXPECT_DOUBLE_EQ(result.residual_norm, 3.);
    EXPECT_EQ(jacobians, 3);
}

TEST(LMLiterature, NielsenIncreasesDampingForPoorModelAgreement)
{
    vector_type x(1);
    x << 0.;
    int                        jacobians = 0;
    levenberg_marquardt_solver solver(
        1,
        1,
        [&](const vector_type&, vector_type& r)
        { r[0] = jacobians == 0 ? 1. : (jacobians == 1 ? .9 : .8); },
        [&](const vector_type&, matrix_type& j)
        {
            j(0, 0) = 1.;
            ++jacobians;
        });
    auto         options     = solver_options_lm_builder()
                                   .with_max_iterations(2)
                                   .with_initial_damping(.1)
                                   .with_geodesic_acceleration(false)
                                   .build();
    const double first_step  = 1. / 1.1;
    const double rho         = (1. - .9 * .9) / (2. * first_step - first_step * first_step);
    const double next_lambda = .1 * std::max(1. / 3., 1. - std::pow(2. * rho - 1., 3.));
    solver.solve(x, *options);
    EXPECT_GT(next_lambda, .1);
    EXPECT_NEAR(x[0], -first_step - .9 / (1. + next_lambda), 1.e-14);
}
