// Behavior of the native Levenberg-Marquardt solver through api::solve.
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

// Each residual component must sit strictly under the function tolerance.
void expect_each_residual_below(const function_type& residuals,
    const vector_type& x,
    index_type         num_residuals,
    double             tolerance)
{
    vector_type r(num_residuals);
    residuals(x, r);
    for (index_type i = 0; i < num_residuals; ++i)
    {
        EXPECT_LT(std::abs(r[i]), tolerance) << "r[" << i << "] = " << r[i];
    }
}

api::least_squares_problem least_squares(std::size_t parameters,
    std::size_t residuals,
    function_type residual,
    jacobian_type jacobian = nullptr)
{
    api::least_squares_problem problem;
    problem.num_parameters = parameters;
    problem.num_residuals  = residuals;
    problem.residuals      = std::move(residual);
    if (jacobian)
    {
        problem.jacobian = std::move(jacobian);
    }
    return problem;
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

// Checks that function tolerance is a strict bound on every residual component.
TEST(LmStopping, FunctionToleranceBoundsEachResidual)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0];
        r[1] = 0.5 * x[0];
    };
    const auto jacobian = [](const vector_type&, matrix_type& j) { j << 1., 0.5; };
    // At x = 1, r = [1, 0.5] and ||r|| = sqrt(1.25). The comparison is strict.
    const double tolerance = std::sqrt(1.25);

    const auto equal = solver_options_lm_builder()
                           .with_geodesic_acceleration(false)
                           .with_function_tolerance(tolerance)
                           .with_gradient_tolerance(0.)
                           .with_parameter_tolerance(0.)
                           .build();
    const auto not_yet =
        api::solve(least_squares(1, 2, residuals, jacobian), vec({1.}), *equal);
    EXPECT_GE(not_yet.iterations, 1u);
    EXPECT_EQ(not_yet.status, api::solver_status::converged);
    expect_each_residual_below(residuals, not_yet.parameters, 2, tolerance);

    const auto above = solver_options_lm_builder()
                           .with_geodesic_acceleration(false)
                           .with_function_tolerance(std::nextafter(tolerance, 2.))
                           .with_gradient_tolerance(0.)
                           .with_parameter_tolerance(0.)
                           .build();
    const auto already =
        api::solve(least_squares(1, 2, residuals, jacobian), vec({1.}), *above);
    EXPECT_EQ(already.iterations, 0u);
    EXPECT_EQ(already.status, api::solver_status::converged);
    EXPECT_EQ(already.parameters[0], 1.);
    expect_each_residual_below(residuals, already.parameters, 2, std::nextafter(tolerance, 2.));
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
        const auto run =
            api::solve(least_squares(2, 2, residuals, jacobian), vec({0., 0.}), *options);
        EXPECT_EQ(run.status, api::solver_status::converged);
        EXPECT_EQ(run.iterations, 0u);
        EXPECT_EQ(run.parameters[0], 0.);
        EXPECT_EQ(run.parameters[1], 0.);
        ASSERT_TRUE(run.residual_norm.has_value());
        EXPECT_NEAR(*run.residual_norm, std::sqrt(5.), 1e-15);
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
    const auto run = api::solve(least_squares(1, 1, residuals, jacobian), vec({0.}), *options);
    EXPECT_EQ(run.status, api::solver_status::converged);
    EXPECT_EQ(run.iterations, 0u);
    EXPECT_EQ(run.parameters[0], 0.);
    ASSERT_TRUE(run.gradient_norm.has_value());
    EXPECT_EQ(*run.gradient_norm, 1.);
}

// Ensures exhausting the iteration budget is reported as max_iterations, not converged.
TEST(LmTermination, IterationBudgetIsNotReportedAsConverged)
{
    const auto options = loose(strategy_enum::NIELSEN, 3).build();
    const auto result  = api::solve(
        least_squares(2, 2, rosenbrock_residual, rosenbrock_jacobian), vec({-1.2, 1.}), *options);
    EXPECT_EQ(result.status, api::solver_status::max_iterations);
    EXPECT_EQ(result.iterations, 3u);
    EXPECT_EQ(result.message.find("converged"), std::string::npos) << result.message;
}

// Checks that a zero iteration budget returns without changing the initial parameters.
TEST(LmTermination, ZeroIterationBudgetLeavesTheStartingPoint)
{
    const auto options = loose(strategy_enum::NIELSEN, 0).build();
    const auto run     = api::solve(
        least_squares(1, 1, parabola_residual, parabola_jacobian), vec({0.1}), *options);
    EXPECT_EQ(run.status, api::solver_status::max_iterations);
    EXPECT_EQ(run.iterations, 0u);
    EXPECT_EQ(run.parameters[0], 0.1);
    ASSERT_TRUE(run.residual_norm.has_value());
    EXPECT_NEAR(*run.residual_norm, 0.99, 1e-14);
}

// Checks that a rejected step at the damping ceiling reports a stall without moving the iterate.
TEST(LmTermination, StallsWhenDampingIsAlreadyAtTheCeiling)
{
    const auto options = loose(strategy_enum::NIELSEN, 50)
                             .with_initial_damping(1e-4)
                             .with_damping_ceiling(1e-4)
                             .build();
    const auto run = api::solve(
        least_squares(1, 1, parabola_residual, parabola_jacobian), vec({0.1}), *options);
    EXPECT_EQ(run.status, api::solver_status::stalled);
    EXPECT_FALSE(run.converged());
    EXPECT_NE(run.message.find("ceiling"), std::string::npos) << run.message;
    EXPECT_EQ(run.accepted_steps, 0u);
    EXPECT_GE(run.rejected_steps, 1u);
    EXPECT_EQ(run.parameters[0], 0.1);
    ASSERT_TRUE(run.residual_norm.has_value());
    EXPECT_NEAR(*run.residual_norm, 0.99, 1e-12);
}

// Ensures non-finite callbacks and a non-finite start throw, and a wrong-sized guess is rejected.
TEST(LmSafeguards, InvalidCallbackOutputsThrow)
{
    const auto options = solver_options_lm_builder().build();
    {
        const auto not_finite = [](const vector_type&, vector_type& r) { r[0] = kNaN; };
        EXPECT_ANY_THROW(api::solve(
            least_squares(1, 1, not_finite, parabola_jacobian), vec({0.1}), *options));
    }
    {
        const auto not_finite = [](const vector_type&, matrix_type& j) { j(0, 0) = kNaN; };
        EXPECT_ANY_THROW(api::solve(
            least_squares(1, 1, parabola_residual, not_finite), vec({0.1}), *options));
    }
    {
        EXPECT_ANY_THROW(api::solve(
            least_squares(1, 1, parabola_residual, parabola_jacobian), vec({kNaN}), *options));
    }
    {
        const auto result = api::solve(
            least_squares(1, 1, parabola_residual, parabola_jacobian), vec({0.1, 0.2}), *options);
        EXPECT_EQ(result.status, api::solver_status::invalid_problem);
        EXPECT_EQ(result.parameters[0], 0.1);
        EXPECT_EQ(result.parameters[1], 0.2);
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
    const auto run =
        api::solve(least_squares(1, 1, residuals, parabola_jacobian), vec({0.1}), *options);
    EXPECT_TRUE(run.converged()) << run.message;
    EXPECT_NEAR(run.parameters[0], 1., 1e-8);
    EXPECT_GE(run.rejected_steps, 1u);
    expect_each_residual_below(residuals, run.parameters, 1, options->function_tolerance());
}

// Ensures relative finite differences reject a zero-valued parameter where scaling is undefined.
TEST(LmSafeguards, RelativeDifferenceAtZeroThrows)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] - 2.;
        r[1] = 3. * x[1] - 9.;
    };
    const auto options = solver_options_lm_builder()
                             .with_geodesic_acceleration(false)
                             .with_finite_difference_step(1e-6)
                             .with_difference_scale(finite_difference_scale::relative)
                             .build();
    EXPECT_ANY_THROW(
        api::solve(least_squares(2, 2, residuals), vec({0., 0.}), *options));
}

// Checks convergence from a nonzero start using both absolute and relative finite differences.
TEST(LmSafeguards, FiniteDifferencesConvergeFromANonzeroStart)
{
    const auto residuals = [](const vector_type& x, vector_type& r)
    {
        r[0] = x[0] - 2.;
        r[1] = 3. * x[1] - 9.;
    };
    for (const auto scale : {finite_difference_scale::absolute, finite_difference_scale::relative})
    {
        const auto options = solver_options_lm_builder()
                                 .with_geodesic_acceleration(false)
                                 .with_max_iterations(100)
                                 .with_function_tolerance(1e-8)
                                 .with_finite_difference_step(1e-6)
                                 .with_difference_scale(scale)
                                 .build();
        const auto run = api::solve(least_squares(2, 2, residuals), vec({0.5, 0.5}), *options);
        EXPECT_TRUE(run.converged()) << run.message;
        EXPECT_NEAR(run.parameters[0], 2., 1e-5);
        EXPECT_NEAR(run.parameters[1], 3., 1e-5);
        expect_each_residual_below(residuals, run.parameters, 2, options->function_tolerance());
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
        const auto run =
            api::solve(least_squares(2, 2, residuals, jacobian), vec({5., 7.}), *options);
        EXPECT_TRUE(run.converged()) << run.message;
        EXPECT_NEAR(run.parameters[0], 1., 1e-8);
        EXPECT_EQ(run.parameters[1], 7.);
        expect_each_residual_below(residuals, run.parameters, 2, options->function_tolerance());
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
    const auto run = api::solve(
        least_squares(2, 2, rosenbrock_residual, rosenbrock_jacobian), vec({-1.2, 1.}), *options);
    EXPECT_EQ(run.status, api::solver_status::converged) << run.message;
    EXPECT_NEAR(run.parameters[0], 1., 1e-6);
    EXPECT_NEAR(run.parameters[1], 1., 1e-6);
    expect_each_residual_below(
        rosenbrock_residual, run.parameters, 2, options->function_tolerance());
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

TEST(LmApi, Solve)
{
    {
        SCOPED_TRACE("ObjectiveIsTheSquaredResidualNorm");
        const auto problem = least_squares(1,
            2,
            [](const vector_type& x, vector_type& r)
            {
                r[0] = x[0] - 1.;
                r[1] = x[0] + 1.;
            },
            [](const vector_type&, matrix_type& j) { j << 1., 1.; });

        const auto lm = api::solve(problem,
            vec({3.}),
            *solver_options_lm_builder().with_gradient_tolerance(1e-8).build());
        EXPECT_TRUE(lm.converged()) << lm.message;
        EXPECT_NEAR(lm.parameters[0], 0., 1e-8);
        EXPECT_NEAR(lm.objective, 2., 1e-12);
        ASSERT_TRUE(lm.residual_norm.has_value());
        EXPECT_NEAR(*lm.residual_norm, std::sqrt(2.), 1e-12);

        const auto gn = api::solve(problem,
            vec({3.}),
            *solver_options_gn_builder().with_gradient_tolerance(1e-8).build());
        EXPECT_NEAR(gn.objective, 2., 1e-12);
    }
    {
        SCOPED_TRACE("BoundsAreRejectedExplicitly");
        auto problem         = least_squares(2, 2, rosenbrock_residual, rosenbrock_jacobian);
        problem.bounds.lower = {-2., -2.};
        problem.bounds.upper = {0.5, 2.};

        const auto result =
            api::solve(problem, vec({-1.2, 1.}), *solver_options_lm_builder().build());
        EXPECT_EQ(result.status, api::solver_status::unsupported_capability);
        EXPECT_NE(result.message.find("bounds"), std::string::npos) << result.message;
        EXPECT_EQ(result.parameters[0], -1.2);
        EXPECT_EQ(result.parameters[1], 1.);
    }
}
