#include <gtest/gtest.h>

#include <cmath>

#include "detail/support.h"
#include "optimization_test_problems.h"
#include "solvers/api/derivative_provider.h"
#include "solvers/api/solve.h"
#include "solvers/integrations/rnc_autodiff.h"

namespace solverslib::api
{
namespace
{
const vector_type kStart = (vector_type(2) << -1.2, 1.0).finished();

struct RosenbrockModel
{
    template <class T> bool operator()(const T* x, T* r) const
    {
        r[0] = T(10.0) * (x[1] - x[0] * x[0]);
        r[1] = T(1.0) - x[0];
        return true;
    }
};

least_squares_problem base_problem()
{
    const auto            tp = testing::make_rosenbrock_problem();
    least_squares_problem problem;
    problem.num_parameters = tp.num_parameters;
    problem.num_residuals  = tp.num_residuals;
    problem.residuals      = tp.residuals;
    return problem;
}

void expect_same_result(const solver_result& a, const solver_result& b)
{
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.backend, b.backend);
    EXPECT_EQ(a.algorithm, b.algorithm);
    EXPECT_EQ(a.effective_derivative_source, b.effective_derivative_source);
    EXPECT_EQ(a.iterations, b.iterations);
    ASSERT_EQ(a.parameters.size(), b.parameters.size());
    EXPECT_LT((a.parameters - b.parameters).norm(), 1e-12);
}

// -- one derivative slot ----------------------------------------------------------
TEST(ProblemApi, EveryDerivativeKindIsAProviderInOneSlot)
{
    const auto tp = testing::make_rosenbrock_problem();

    auto problem = base_problem();
    EXPECT_EQ(problem.derivative_provider(), nullptr);

    problem.set_jacobian(tp.jacobian);
    ASSERT_NE(problem.derivative_provider(), nullptr);
    EXPECT_EQ(problem.derivative_provider()->source(), derivative_mode::supplied);
    EXPECT_TRUE(problem.has_callable_jacobian());

    problem.set_jacobian_provider(finite_difference(tp.residuals, 2, 2));
    EXPECT_EQ(problem.derivative_provider()->source(), derivative_mode::finite_difference);
    EXPECT_FALSE(problem.has_callable_jacobian());

    problem.set_curve_derivatives(
        [](const vector_type&, const std::vector<vector_type>&, int, rnc_curve_derivatives&) {},
        derivative_mode::supplied);
    EXPECT_TRUE(static_cast<bool>(problem.derivative_provider()->curve_derivatives()));

    problem.set_jacobian_provider(nullptr);
    EXPECT_EQ(problem.derivative_provider(), nullptr);
}

TEST(ProblemApi, SetJacobianNeedsResidualsFirst)
{
    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 1;
    EXPECT_THROW(
        problem.set_jacobian([](const vector_type&, matrix_type&) {}), std::invalid_argument);
}

TEST(ProblemApi, CurveProviderServesAsOrdinaryJacobianForOtherBackends)
{
    auto problem = rnc_least_squares(RosenbrockModel{}, 2, 2);

    solve_options options;
    options.backend   = backend::native;
    options.algorithm = algorithm::levenberg_marquardt;
    const auto result = solve(problem, kStart, options);
    EXPECT_TRUE(result.converged()) << result.message;
    ASSERT_TRUE(result.effective_derivative_source.has_value());
    EXPECT_EQ(*result.effective_derivative_source, derivative_mode::automatic_differentiation);
    EXPECT_NEAR(result.parameters[0], 1.0, 1e-6);
}

// -- legacy fields keep working for one release ----------------------------------------
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
TEST(ProblemApi, LegacyJacobianCallbackDispatchesLikeSetJacobian)
{
    const auto tp = testing::make_rosenbrock_problem();

    auto legacy     = base_problem();
    legacy.jacobian = tp.jacobian;
    auto modern     = base_problem();
    modern.set_jacobian(tp.jacobian);

    EXPECT_EQ(inspect(legacy).has_jacobian, inspect(modern).has_jacobian);
    EXPECT_EQ(inspect(legacy).has_callable_jacobian, inspect(modern).has_callable_jacobian);
    expect_same_result(solve(legacy, kStart), solve(modern, kStart));
}

TEST(ProblemApi, LegacyProviderFieldDispatchesLikeSetProvider)
{
    const auto tp       = testing::make_rosenbrock_problem();
    auto       provider = analytic_jacobian(tp.residuals, tp.jacobian, 2, 2);

    auto legacy              = base_problem();
    legacy.jacobian_provider = provider;
    auto modern              = base_problem();
    modern.set_jacobian_provider(provider);

    expect_same_result(solve(legacy, kStart), solve(modern, kStart));
}

TEST(ProblemApi, ModernSlotWinsOverLegacyFields)
{
    const auto tp = testing::make_rosenbrock_problem();

    auto problem     = base_problem();
    problem.jacobian = tp.jacobian;  // legacy: supplied
    problem.set_jacobian_provider(finite_difference(tp.residuals, 2, 2));
    const auto result = solve(problem, kStart);
    ASSERT_TRUE(result.effective_derivative_source.has_value());
    EXPECT_EQ(*result.effective_derivative_source, derivative_mode::finite_difference);
}

TEST(ProblemApi, LegacyGradientFieldsDispatchLikeSetters)
{
    const auto gradient = [](const vector_type& x, vector_type& g)
    { g = 2.0 * (x - vector_type::Constant(2, 2.0)); };
    const auto objective = [](const vector_type& x)
    { return (x - vector_type::Constant(2, 2.0)).squaredNorm(); };

    optimization_problem legacy;
    legacy.num_parameters = 2;
    legacy.objective      = objective;
    legacy.gradient       = gradient;

    optimization_problem legacy_provider;
    legacy_provider.num_parameters    = 2;
    legacy_provider.objective         = objective;
    legacy_provider.gradient_provider = analytic_gradient(gradient, 2);

    optimization_problem modern;
    modern.num_parameters = 2;
    modern.objective      = objective;
    modern.set_gradient(gradient);

    const auto expected = solve(modern, vector_type::Zero(2));
    EXPECT_TRUE(expected.converged());
    expect_same_result(solve(legacy, vector_type::Zero(2)), expected);
    expect_same_result(solve(legacy_provider, vector_type::Zero(2)), expected);
}
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// -- solve_options::lm ----------------------------------------------------------------------
TEST(ProblemApi, LmOptionsReachTheNativeKernel)
{
    const auto tp      = testing::make_rosenbrock_problem();
    auto       problem = base_problem();
    problem.set_jacobian(tp.jacobian);

    solve_options options;
    options.backend   = backend::native;
    options.algorithm = algorithm::levenberg_marquardt;

    // Without geodesic acceleration each iteration costs one residual evaluation
    // plus the initial one; with it (the default) each costs two.
    options.lm                        = lm_options{};
    options.lm->geodesic_acceleration = false;
    const auto plain                  = solve(problem, kStart, options);
    ASSERT_TRUE(plain.converged()) << plain.message;
    ASSERT_TRUE(plain.residual_evaluations.has_value());
    EXPECT_EQ(*plain.residual_evaluations, 1 + plain.iterations);

    options.lm->geodesic_acceleration = true;
    const auto accelerated            = solve(problem, kStart, options);
    ASSERT_TRUE(accelerated.converged()) << accelerated.message;
    EXPECT_EQ(*accelerated.residual_evaluations, 1 + 2 * accelerated.iterations);

    // The variant is selectable too, and still converges on this problem.
    options.lm->variant = lm_variant::levenberg_marquardt;
    EXPECT_TRUE(solve(problem, kStart, options).converged());
}

TEST(ProblemApi, InvalidLmOptionsAreRejectedAtTheBoundary)
{
    const auto tp      = testing::make_rosenbrock_problem();
    auto       problem = base_problem();
    problem.set_jacobian(tp.jacobian);

    solve_options options;
    options.lm                  = lm_options{};
    options.lm->initial_damping = -1.0;
    EXPECT_EQ(solve(problem, kStart, options).status, solver_status::invalid_problem);

    options.lm                          = lm_options{};
    options.lm->damping_increase_factor = 0.5;
    EXPECT_EQ(solve(problem, kStart, options).status, solver_status::invalid_problem);

    options.lm                = lm_options{};
    options.lm->damping_floor = 1e20;  // above the ceilings
    EXPECT_EQ(solve(problem, kStart, options).status, solver_status::invalid_problem);
}
}  // namespace
}  // namespace solverslib::api
