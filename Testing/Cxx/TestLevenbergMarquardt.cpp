// Behavior of the native Levenberg-Marquardt solver through its public result.
// The kernel does not publish per-trial traces, so these checks use the
// returned iterate, status, and step counts.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include "api/dispatch.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_lm.h"
#include "solvers/levenberg_marquardt_solver.h"

using namespace solverslib;

namespace
{
using strategy_enum = levenberg_marquardt_solver_enum;
using linear_enum   = levenberg_marquardt_linear_solver_enum;

const double kNaN = std::numeric_limits<double>::quiet_NaN();

vector_type vec(std::initializer_list<double> values)
{
    vector_type v(static_cast<index_type>(values.size()));
    index_type  i = 0;
    for (double value : values)
    {
        v[i++] = value;
    }
    return v;
}

solver_options_lm_builder loose(strategy_enum strategy, int max_iterations)
{
    return solver_options_lm_builder()
        .with_type(strategy)
        .with_geodesic_acceleration(false)
        .with_max_iterations(max_iterations)
        .with_function_tolerance(0.)
        .with_gradient_tolerance(0.)
        .with_parameter_tolerance(0.);
}

void parabola_residual(const vector_type& x, vector_type& r)
{
    r[0] = x[0] * x[0] - 1.;
}
void parabola_jacobian(const vector_type& x, matrix_type& j)
{
    j(0, 0) = 2. * x[0];
}

void rosenbrock_residual(const vector_type& x, vector_type& r)
{
    r[0] = 10. * (x[1] - x[0] * x[0]);
    r[1] = 1. - x[0];
}
void rosenbrock_jacobian(const vector_type& x, matrix_type& j)
{
    j << -20. * x[0], 10., -1., 0.;
}
}  // namespace

// Documents the default strategy, linear solver, damping, scaling, and acceleration settings.
TEST(LmOptions, DefaultsMatchTheSources)
{
    const auto options = solver_options_lm_builder().build();
    EXPECT_EQ(options->type(), strategy_enum::NIELSEN);
    EXPECT_EQ(options->linear_solver(), linear_enum::NORMAL_LDLT);
    EXPECT_EQ(options->initial_damping(), 1e-4);
    EXPECT_EQ(options->initial_rejection_multiplier(), 2.);
    EXPECT_EQ(options->damping_decrease_factor(), 9.);
    EXPECT_EQ(options->damping_increase_factor(), 11.);
    EXPECT_EQ(options->damping_floor(), 1e-7);
    EXPECT_EQ(options->nielsen_damping_floor(), 1e-15);
    EXPECT_EQ(options->damping_ceiling(), 1e12);
    EXPECT_EQ(options->levenberg_marquardt_damping_ceiling(), 1e7);
    EXPECT_EQ(options->diagonal_scaling_floor(), 1e-12);
    EXPECT_EQ(options->geodesic_acceleration_step(), 0.05);
    EXPECT_EQ(options->geodesic_acceleration_threshold(), 0.75);
    EXPECT_EQ(options->roundoff_noise_factor(), 8.);
    EXPECT_EQ(options->bold_acceptance_exponent(), 2.);
    EXPECT_FALSE(options->bold_acceptance());
    EXPECT_TRUE(options->geodesic_acceleration());
    EXPECT_EQ(options->finite_difference_step(), 1e-5);
    EXPECT_EQ(options->difference_scale(), finite_difference_scale::absolute);
}

// Ensures invalid damping, acceleration, tolerance, and iteration settings are rejected.
TEST(LmOptions, RejectsInvalidValues)
{
    EXPECT_THROW(
        solver_options_lm_builder().with_initial_damping(0.).build(), std::invalid_argument);
    EXPECT_THROW(
        solver_options_lm_builder().with_initial_damping(kNaN).build(), std::invalid_argument);
    EXPECT_THROW(solver_options_lm_builder().with_geodesic_acceleration_step(0.).build(),
        std::invalid_argument);
    EXPECT_THROW(solver_options_lm_builder().with_initial_rejection_multiplier(1.).build(),
        std::invalid_argument);
    EXPECT_THROW(solver_options_lm_builder().with_damping_increase_factor(1.).build(),
        std::invalid_argument);
    EXPECT_THROW(
        solver_options_lm_builder().with_function_tolerance(-1.).build(), std::invalid_argument);
    EXPECT_THROW(
        solver_options_lm_builder().with_max_iterations(-1).build(), std::invalid_argument);
}

// Checks that function tolerance uses half the squared residual norm and honors its strict bound.
TEST(LmStopping, FunctionToleranceIsHalfTheSquaredResidual)
{
    const auto residuals = [](const vector_type& x, vector_type& r) { r[0] = x[0]; };
    const auto jacobian  = [](const vector_type&, matrix_type& j) { j(0, 0) = 1.; };
    // ||r|| = 0.5, so 0.5 * ||r||^2 = 0.125. The comparison is strict.
    const double half_squared = 0.125;

    const auto equal = solver_options_lm_builder()
                           .with_geodesic_acceleration(false)
                           .with_function_tolerance(half_squared)
                           .with_gradient_tolerance(0.)
                           .with_parameter_tolerance(0.)
                           .build();
    vector_type                not_yet_x = vec({0.5});
    levenberg_marquardt_solver solver(1, 1, residuals, jacobian);
    const auto                 not_yet = solver.solve(not_yet_x, *equal);
    EXPECT_GE(not_yet.iterations, 1u);
    EXPECT_EQ(not_yet.status, native_convergence::function_converged);
    EXPECT_LT(0.5 * not_yet.residual_norm * not_yet.residual_norm, half_squared);

    const auto above = solver_options_lm_builder()
                           .with_geodesic_acceleration(false)
                           .with_function_tolerance(std::nextafter(half_squared, 1.))
                           .with_gradient_tolerance(0.)
                           .with_parameter_tolerance(0.)
                           .build();
    vector_type already_x = vec({0.5});
    const auto  already   = solver.solve(already_x, *above);
    EXPECT_EQ(already.iterations, 0u);
    EXPECT_EQ(already.status, native_convergence::function_converged);
    EXPECT_EQ(already_x[0], 0.5);
    EXPECT_EQ(already.residual_norm, 0.5);
}

// Checks that all damping strategies stop at a stationary point even when residuals remain.
TEST(LmStopping, StationaryStartWithNonzeroResidual)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] * x[0] + 1.;
        r[1] = x[1] * x[1] + 2.;
    };
    const auto jacobian = [](const vector_type& x, matrix_type& j)
    { j << 2. * x[0], 0., 0., 2. * x[1]; };

    for (const auto strategy : {strategy_enum::LEVENBERG_MARQUARDT,
             strategy_enum::QUADRATIC_INTERPOLATION,
             strategy_enum::NIELSEN})
    {
        const auto options =
            solver_options_lm_builder().with_type(strategy).with_gradient_tolerance(0.).build();
        vector_type x   = vec({0., 0.});
        const auto  run = levenberg_marquardt_solver(2, 2, residuals, jacobian).solve(x, *options);
        EXPECT_EQ(run.status, native_convergence::gradient_converged);
        EXPECT_EQ(run.iterations, 0u);
        EXPECT_EQ(x[0], 0.);
        EXPECT_EQ(x[1], 0.);
        EXPECT_NEAR(run.residual_norm, std::sqrt(5.), 1e-15);
        ASSERT_TRUE(run.gradient_norm.has_value());
        EXPECT_EQ(*run.gradient_norm, 0.);
    }
}

// Checks gradient convergence against the norm of J-transpose times the residual.
TEST(LmStopping, GradientToleranceBoundsJtR)
{
    const auto residuals = [](const vector_type& x, vector_type& r) { r[0] = x[0] - 1.; };
    const auto jacobian  = [](const vector_type&, matrix_type& j) { j(0, 0) = 1.; };
    const auto options   = solver_options_lm_builder()
                             .with_function_tolerance(0.)
                             .with_gradient_tolerance(1.5)
                             .build();
    vector_type x   = vec({0.});
    const auto  run = levenberg_marquardt_solver(1, 1, residuals, jacobian).solve(x, *options);
    EXPECT_EQ(run.status, native_convergence::gradient_converged);
    EXPECT_EQ(run.iterations, 0u);
    EXPECT_EQ(x[0], 0.);
    ASSERT_TRUE(run.gradient_norm.has_value());
    EXPECT_EQ(*run.gradient_norm, 1.);
}

// Ensures exhausting the iteration budget stays non-converged in both native and public API results.
TEST(LmTermination, IterationBudgetIsNotReportedAsConverged)
{
    const auto  options = loose(strategy_enum::NIELSEN, 3).build();
    vector_type x       = vec({-1.2, 1.});
    const auto  run     = levenberg_marquardt_solver(2, 2, rosenbrock_residual, rosenbrock_jacobian)
                         .solve(x, *options);
    EXPECT_EQ(run.status, native_convergence::not_converged);
    EXPECT_EQ(run.iterations, 3u);
    EXPECT_EQ(run.message.find("converged"), std::string::npos) << run.message;

    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 2;
    problem.residuals      = rosenbrock_residual;
    problem.jacobian       = rosenbrock_jacobian;
    const auto result      = api::solve(problem, vec({-1.2, 1.}), *options);
    EXPECT_EQ(result.status, api::solver_status::max_iterations);
    EXPECT_EQ(result.iterations, 3u);
    EXPECT_EQ(result.message.find("converged"), std::string::npos) << result.message;
}

// Checks that a zero iteration budget returns without changing the initial parameters.
TEST(LmTermination, ZeroIterationBudgetLeavesTheStartingPoint)
{
    const auto  options = loose(strategy_enum::NIELSEN, 0).build();
    vector_type x       = vec({0.1});
    const auto  run =
        levenberg_marquardt_solver(1, 1, parabola_residual, parabola_jacobian).solve(x, *options);
    EXPECT_EQ(run.status, native_convergence::not_converged);
    EXPECT_EQ(run.iterations, 0u);
    EXPECT_EQ(x[0], 0.1);
    EXPECT_NEAR(run.residual_norm, 0.99, 1e-14);
}

// Checks that a rejected step at the damping ceiling reports a stall without moving the iterate.
TEST(LmTermination, StallsWhenDampingIsAlreadyAtTheCeiling)
{
    const auto options = loose(strategy_enum::NIELSEN, 50)
                             .with_initial_damping(1e-4)
                             .with_damping_ceiling(1e-4)
                             .build();
    vector_type x = vec({0.1});
    const auto  run =
        levenberg_marquardt_solver(1, 1, parabola_residual, parabola_jacobian).solve(x, *options);
    EXPECT_EQ(run.status, native_convergence::stalled);
    EXPECT_FALSE(run.converged());
    EXPECT_NE(run.message.find("ceiling"), std::string::npos) << run.message;
    EXPECT_EQ(run.accepted_steps, 0u);
    EXPECT_GE(run.rejected_steps, 1u);
    EXPECT_EQ(x[0], 0.1);
    EXPECT_NEAR(run.residual_norm, 0.99, 1e-12);
}

// Ensures non-finite callback values, non-finite initial parameters, and wrong dimensions throw.
TEST(LmSafeguards, InvalidCallbackOutputsThrow)
{
    const auto options = solver_options_lm_builder().build();
    {
        const auto  not_finite = [](const vector_type&, vector_type& r) { r[0] = kNaN; };
        vector_type x          = vec({0.1});
        EXPECT_ANY_THROW(
            levenberg_marquardt_solver(1, 1, not_finite, parabola_jacobian).solve(x, *options));
    }
    {
        const auto  not_finite = [](const vector_type&, matrix_type& j) { j(0, 0) = kNaN; };
        vector_type x          = vec({0.1});
        EXPECT_ANY_THROW(
            levenberg_marquardt_solver(1, 1, parabola_residual, not_finite).solve(x, *options));
    }
    {
        vector_type x = vec({kNaN});
        EXPECT_ANY_THROW(levenberg_marquardt_solver(1, 1, parabola_residual, parabola_jacobian)
                .solve(x, *options));
    }
    {
        vector_type x = vec({0.1, 0.2});
        EXPECT_ANY_THROW(levenberg_marquardt_solver(1, 1, parabola_residual, parabola_jacobian)
                .solve(x, *options));
    }
}

// Checks that a non-finite trial is rejected while subsequent valid steps can still converge.
TEST(LmSafeguards, NonFiniteTrialIsRejected)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    { r[0] = x[0] < 2. ? x[0] * x[0] - 1. : kNaN; };
    const auto options = solver_options_lm_builder()
                             .with_geodesic_acceleration(false)
                             .with_max_iterations(200)
                             .with_function_tolerance(1e-20)
                             .with_gradient_tolerance(0.)
                             .with_parameter_tolerance(0.)
                             .build();
    vector_type x = vec({0.1});
    const auto  run =
        levenberg_marquardt_solver(1, 1, residuals, parabola_jacobian).solve(x, *options);
    EXPECT_TRUE(run.converged()) << run.message;
    EXPECT_NEAR(x[0], 1., 1e-8);
    EXPECT_GE(run.rejected_steps, 1u);
}

// Ensures relative finite differences reject a zero-valued parameter where scaling is undefined.
TEST(LmSafeguards, RelativeDifferenceAtZeroThrows)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] - 2.;
        r[1] = 3. * x[1] - 9.;
    };
    const auto  options = solver_options_lm_builder().with_geodesic_acceleration(false).build();
    vector_type x       = vec({0., 0.});
    EXPECT_ANY_THROW(levenberg_marquardt_solver(
        2, 2, residuals, nullptr, 1e-6, finite_difference_scale::relative)
            .solve(x, *options));
}

// Checks convergence from a nonzero start using both absolute and relative finite differences.
TEST(LmSafeguards, FiniteDifferencesConvergeFromANonzeroStart)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] - 2.;
        r[1] = 3. * x[1] - 9.;
    };
    const auto options = solver_options_lm_builder()
                             .with_geodesic_acceleration(false)
                             .with_max_iterations(100)
                             .with_function_tolerance(1e-8)
                             .build();
    for (const auto scale : {finite_difference_scale::absolute, finite_difference_scale::relative})
    {
        vector_type x = vec({0.5, 0.5});
        const auto  run =
            levenberg_marquardt_solver(2, 2, residuals, nullptr, 1e-6, scale).solve(x, *options);
        EXPECT_TRUE(run.converged()) << run.message;
        EXPECT_NEAR(x[0], 2., 1e-5);
        EXPECT_NEAR(x[1], 3., 1e-5);
    }
}

// Ensures parameters with inactive Jacobian columns retain their initial values for each strategy.
TEST(LmSafeguards, InactiveColumnStaysPut)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] - 1.;
        r[1] = 2. * (x[0] - 1.);
    };
    const auto jacobian = [](const vector_type&, matrix_type& j) { j << 1., 0., 2., 0.; };
    for (const auto strategy : {strategy_enum::LEVENBERG_MARQUARDT,
             strategy_enum::QUADRATIC_INTERPOLATION,
             strategy_enum::NIELSEN})
    {
        const auto options = solver_options_lm_builder()
                                 .with_type(strategy)
                                 .with_geodesic_acceleration(false)
                                 .with_max_iterations(200)
                                 .with_function_tolerance(1e-20)
                                 .with_gradient_tolerance(0.)
                                 .with_parameter_tolerance(0.)
                                 .build();
        vector_type x   = vec({5., 7.});
        const auto  run = levenberg_marquardt_solver(2, 2, residuals, jacobian).solve(x, *options);
        EXPECT_TRUE(run.converged()) << run.message;
        EXPECT_NEAR(x[0], 1., 1e-8);
        EXPECT_EQ(x[1], 7.);
    }
}

class LmCombination
    : public ::testing::TestWithParam<std::tuple<strategy_enum, bool, bool, linear_enum>>
{
};

// Checks Rosenbrock convergence across strategies, extensions, and linear solver backends.
TEST_P(LmCombination, ConvergesOnRosenbrock)
{
    const auto [strategy, geodesic, bold, linear] = GetParam();
    const auto options                            = solver_options_lm_builder()
                             .with_type(strategy)
                             .with_geodesic_acceleration(geodesic)
                             .with_bold_acceptance(bold)
                             .with_linear_solver(linear)
                             .with_max_iterations(500)
                             .with_function_tolerance(1e-20)
                             .with_gradient_tolerance(0.)
                             .with_parameter_tolerance(0.)
                             .build();
    vector_type x   = vec({-1.2, 1.});
    const auto  run = levenberg_marquardt_solver(2, 2, rosenbrock_residual, rosenbrock_jacobian)
                         .solve(x, *options);
    EXPECT_EQ(run.status, native_convergence::function_converged) << run.message;
    EXPECT_NEAR(x[0], 1., 1e-6);
    EXPECT_NEAR(x[1], 1., 1e-6);
    EXPECT_LT(0.5 * run.residual_norm * run.residual_norm, 1e-20);
    ASSERT_TRUE(run.accepted_steps.has_value());
    EXPECT_GE(*run.accepted_steps, 1u);
}

INSTANTIATE_TEST_SUITE_P(AllStrategiesAndExtensions,
    LmCombination,
    ::testing::Combine(::testing::Values(strategy_enum::LEVENBERG_MARQUARDT,
                           strategy_enum::QUADRATIC_INTERPOLATION,
                           strategy_enum::NIELSEN),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Values(linear_enum::NORMAL_LDLT, linear_enum::AUGMENTED_QR)));

// Verifies the public least-squares objective is half the squared residual norm for LM and GN.
TEST(LmApi, ObjectiveIsTheSquaredResidualNorm)
{
    api::least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 2;
    problem.residuals      = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] - 1.;
        r[1] = x[0] + 1.;
    };
    problem.jacobian = [](const vector_type&, matrix_type& j) { j << 1., 1.; };

    const auto lm = api::solve(
        problem, vec({3.}), *solver_options_lm_builder().with_gradient_tolerance(1e-8).build());
    EXPECT_TRUE(lm.converged()) << lm.message;
    EXPECT_NEAR(lm.parameters[0], 0., 1e-8);
    EXPECT_NEAR(lm.objective, 2., 1e-12);
    ASSERT_TRUE(lm.residual_norm.has_value());
    EXPECT_NEAR(*lm.residual_norm, std::sqrt(2.), 1e-12);

    const auto gn = api::solve(
        problem, vec({3.}), *solver_options_gn_builder().with_gradient_tolerance(1e-8).build());
    EXPECT_NEAR(gn.objective, 2., 1e-12);
}

// Ensures the API reports bounds as unsupported and preserves the supplied starting parameters.
TEST(LmApi, BoundsAreRejectedExplicitly)
{
    api::least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 2;
    problem.residuals      = rosenbrock_residual;
    problem.jacobian       = rosenbrock_jacobian;
    problem.bounds.lower   = {-2., -2.};
    problem.bounds.upper   = {0.5, 2.};

    const auto result = api::solve(problem, vec({-1.2, 1.}), *solver_options_lm_builder().build());
    EXPECT_EQ(result.status, api::solver_status::unsupported_capability);
    EXPECT_NE(result.message.find("bounds"), std::string::npos) << result.message;
    EXPECT_EQ(result.parameters[0], -1.2);
    EXPECT_EQ(result.parameters[1], 1.);
}
