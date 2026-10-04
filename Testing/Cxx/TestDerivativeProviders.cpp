#include <cmath>
#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "detail/support.h"
#include "optimization_test_problems.h"
#include "solvers/api/derivative_provider.h"

// Solver API for end-to-end integration tests
#include "solvers/api/problem.h"
#include "solvers/api/solve.h"

#if SOLVERS_HAS_CERES
#include "solvers/integrations/autodiff_provider.h"
#endif

namespace solverslib
{
namespace
{

using api::analytic_gradient;
using api::analytic_jacobian;
using api::AnalyticGradientProvider;
using api::AnalyticJacobianProvider;
using api::check_gradient;
using api::check_jacobian;
using api::finite_difference;
using api::finite_difference_gradient;
using api::FiniteDifferenceGradientProvider;
using api::FiniteDifferenceJacobianProvider;
using testing::make_rosenbrock_problem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vector_type rosenbrock_x()
{
    vector_type x = make_vector(2);
    x[0]          = 1.5;
    x[1]          = 0.5;
    return x;
}

// ---------------------------------------------------------------------------
// AnalyticJacobianProvider
// ---------------------------------------------------------------------------

TEST(AnalyticJacobianProvider, ComputeResiduals)
{
    const auto& tp = make_rosenbrock_problem();
    auto        provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    matrix_type       J = make_matrix(tp.num_residuals, tp.num_parameters);
    provider->compute(x, r, J);

    vector_type r_expected = make_vector(tp.num_residuals);
    tp.residuals(x, r_expected);

    EXPECT_NEAR(r[0], r_expected[0], 1e-14);
    EXPECT_NEAR(r[1], r_expected[1], 1e-14);
}

TEST(AnalyticJacobianProvider, ComputeJacobian)
{
    const auto& tp = make_rosenbrock_problem();
    auto        provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    matrix_type       J = make_matrix(tp.num_residuals, tp.num_parameters);
    provider->compute(x, r, J);

    matrix_type J_expected = make_matrix(tp.num_residuals, tp.num_parameters);
    tp.jacobian(x, J_expected);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(J(i, j), J_expected(i, j), 1e-14)
                << "Jacobian mismatch at (" << i << "," << j << ")";
}

TEST(AnalyticJacobianProvider, ResidualsOnly)
{
    const auto& tp = make_rosenbrock_problem();
    auto        provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    provider->residuals_only(x, r);

    vector_type r_expected = make_vector(tp.num_residuals);
    tp.residuals(x, r_expected);

    EXPECT_NEAR(r[0], r_expected[0], 1e-14);
    EXPECT_NEAR(r[1], r_expected[1], 1e-14);
}

TEST(AnalyticJacobianProvider, Metadata)
{
    const auto& tp = make_rosenbrock_problem();
    auto        provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    EXPECT_EQ(provider->num_parameters(), tp.num_parameters);
    EXPECT_EQ(provider->num_residuals(), tp.num_residuals);
    EXPECT_EQ(provider->source(), api::derivative_mode::supplied);
    EXPECT_EQ(provider->ceres_factory(), nullptr);
}

TEST(AnalyticJacobianProvider, NullResidualThrows)
{
    const auto& tp = make_rosenbrock_problem();
    EXPECT_ANY_THROW(
        AnalyticJacobianProvider(nullptr, tp.jacobian, tp.num_parameters, tp.num_residuals));
}

TEST(AnalyticJacobianProvider, NullJacobianThrows)
{
    const auto& tp = make_rosenbrock_problem();
    EXPECT_ANY_THROW(
        AnalyticJacobianProvider(tp.residuals, nullptr, tp.num_parameters, tp.num_residuals));
}

// ---------------------------------------------------------------------------
// FiniteDifferenceJacobianProvider
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceJacobianProvider, ComputeResiduals)
{
    const auto& tp       = make_rosenbrock_problem();
    auto        provider = finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    matrix_type       J = make_matrix(tp.num_residuals, tp.num_parameters);
    provider->compute(x, r, J);

    vector_type r_expected = make_vector(tp.num_residuals);
    tp.residuals(x, r_expected);

    EXPECT_NEAR(r[0], r_expected[0], 1e-14);
    EXPECT_NEAR(r[1], r_expected[1], 1e-14);
}

TEST(FiniteDifferenceJacobianProvider, JacobianAccuracy)
{
    const auto& tp       = make_rosenbrock_problem();
    auto        provider = finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    matrix_type       J = make_matrix(tp.num_residuals, tp.num_parameters);
    provider->compute(x, r, J);

    matrix_type J_expected = make_matrix(tp.num_residuals, tp.num_parameters);
    tp.jacobian(x, J_expected);

    // FD with central differences achieves ~1e-6 relative error
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(J(i, j), J_expected(i, j), 1e-4)
                << "FD Jacobian mismatch at (" << i << "," << j << ")";
}

TEST(FiniteDifferenceJacobianProvider, ResidualsOnly)
{
    const auto& tp       = make_rosenbrock_problem();
    auto        provider = finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    provider->residuals_only(x, r);

    vector_type r_expected = make_vector(tp.num_residuals);
    tp.residuals(x, r_expected);

    EXPECT_NEAR(r[0], r_expected[0], 1e-14);
    EXPECT_NEAR(r[1], r_expected[1], 1e-14);
}

TEST(FiniteDifferenceJacobianProvider, Metadata)
{
    const auto& tp       = make_rosenbrock_problem();
    auto        provider = finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals);

    EXPECT_EQ(provider->num_parameters(), tp.num_parameters);
    EXPECT_EQ(provider->num_residuals(), tp.num_residuals);
    EXPECT_EQ(provider->source(), api::derivative_mode::finite_difference);
}

TEST(FiniteDifferenceJacobianProvider, NonPositiveStepThrows)
{
    const auto& tp = make_rosenbrock_problem();
    EXPECT_ANY_THROW(
        FiniteDifferenceJacobianProvider(tp.residuals, tp.num_parameters, tp.num_residuals, -1e-7));
    EXPECT_ANY_THROW(
        FiniteDifferenceJacobianProvider(tp.residuals, tp.num_parameters, tp.num_residuals, 0.0));
}

// ---------------------------------------------------------------------------
// AnalyticGradientProvider
// ---------------------------------------------------------------------------

TEST(AnalyticGradientProvider, Compute)
{
    // f(x) = x[0]^2 + 2*x[1]^2, g = [2*x[0], 4*x[1]]
    auto gf = [](const vector_type& x, vector_type& g)
    {
        g[0] = 2.0 * x[0];
        g[1] = 4.0 * x[1];
    };
    auto provider = analytic_gradient(gf, 2);

    vector_type x = make_vector(2);
    x[0]          = 3.0;
    x[1]          = 1.5;
    vector_type g = make_vector(2);
    provider->compute(x, g);

    EXPECT_NEAR(g[0], 6.0, 1e-14);
    EXPECT_NEAR(g[1], 6.0, 1e-14);
}

TEST(AnalyticGradientProvider, Metadata)
{
    auto gf       = [](const vector_type& x, vector_type& g) { g[0] = x[0]; };
    auto provider = analytic_gradient(gf, 1);

    EXPECT_EQ(provider->num_parameters(), 1u);
    EXPECT_EQ(provider->source(), api::derivative_mode::supplied);
}

TEST(AnalyticGradientProvider, NullGradientThrows)
{
    EXPECT_ANY_THROW(AnalyticGradientProvider(nullptr, 2));
}

// ---------------------------------------------------------------------------
// FiniteDifferenceGradientProvider
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceGradientProvider, GradientAccuracy)
{
    // f(x) = x[0]^2 + 2*x[1]^2
    auto of       = [](const vector_type& x) { return x[0] * x[0] + 2.0 * x[1] * x[1]; };
    auto provider = finite_difference_gradient(of, 2);

    vector_type x = make_vector(2);
    x[0]          = 3.0;
    x[1]          = 1.5;
    vector_type g = make_vector(2);
    provider->compute(x, g);

    EXPECT_NEAR(g[0], 6.0, 1e-5);
    EXPECT_NEAR(g[1], 6.0, 1e-5);
}

TEST(FiniteDifferenceGradientProvider, Metadata)
{
    auto of       = [](const vector_type& x) { return x[0]; };
    auto provider = finite_difference_gradient(of, 1);

    EXPECT_EQ(provider->num_parameters(), 1u);
    EXPECT_EQ(provider->source(), api::derivative_mode::finite_difference);
}

TEST(FiniteDifferenceGradientProvider, NonPositiveStepThrows)
{
    auto of = [](const vector_type& x) { return x[0]; };
    EXPECT_ANY_THROW(FiniteDifferenceGradientProvider(of, 1, -1e-7));
    EXPECT_ANY_THROW(FiniteDifferenceGradientProvider(of, 1, 0.0));
}

// ---------------------------------------------------------------------------
// check_jacobian — cross-provider validation utility
// ---------------------------------------------------------------------------

TEST(CheckJacobian, AnalyticVsFiniteDifference)
{
    const auto& tp = make_rosenbrock_problem();

    auto analytic =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);
    auto fd = finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals);

    const vector_type x      = rosenbrock_x();
    auto              result = check_jacobian(*analytic, *fd, x, 1e-3);

    EXPECT_TRUE(result.passed) << result.summary;
    EXPECT_GT(result.max_abs_error, 0.0);   // FD != analytic exactly
    EXPECT_LT(result.max_abs_error, 1e-3);  // but within FD tolerance
}

TEST(CheckJacobian, IdenticalProvidersPasses)
{
    const auto& tp = make_rosenbrock_problem();
    auto a = analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);
    auto b = analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    const vector_type x      = rosenbrock_x();
    auto              result = check_jacobian(*a, *b, x);

    EXPECT_TRUE(result.passed);
    EXPECT_NEAR(result.max_abs_error, 0.0, 1e-14);
}

TEST(CheckJacobian, SummaryContainsPassFail)
{
    const auto& tp = make_rosenbrock_problem();
    auto a = analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);
    auto b = analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    const vector_type x    = rosenbrock_x();
    auto              pass = check_jacobian(*a, *b, x);
    EXPECT_NE(pass.summary.find("PASS"), std::string::npos);

    // Provide a deliberately wrong Jacobian to trigger FAIL
    auto wrong_jacobian = [](const vector_type&, matrix_type& J) { J.setConstant(999.0); };
    auto c = analytic_jacobian(tp.residuals, wrong_jacobian, tp.num_parameters, tp.num_residuals);
    auto fail = check_jacobian(*a, *c, x);
    EXPECT_FALSE(fail.passed);
    EXPECT_NE(fail.summary.find("FAIL"), std::string::npos);
}

// ---------------------------------------------------------------------------
// check_gradient — cross-provider validation utility
// ---------------------------------------------------------------------------

TEST(CheckGradient, AnalyticVsFiniteDifference)
{
    auto of = [](const vector_type& x) { return x[0] * x[0] + 2.0 * x[1] * x[1]; };
    auto gf = [](const vector_type& x, vector_type& g)
    {
        g[0] = 2.0 * x[0];
        g[1] = 4.0 * x[1];
    };
    auto analytic = analytic_gradient(gf, 2);
    auto fd       = finite_difference_gradient(of, 2);

    vector_type x = make_vector(2);
    x[0]          = 3.0;
    x[1]          = 1.5;
    auto result   = check_gradient(*analytic, *fd, x, 1e-3);

    EXPECT_TRUE(result.passed) << result.summary;
}

TEST(CheckGradient, IdenticalProvidersPasses)
{
    auto gf = [](const vector_type& x, vector_type& g) { g[0] = 2.0 * x[0]; };
    auto a  = analytic_gradient(gf, 1);
    auto b  = analytic_gradient(gf, 1);

    vector_type x = make_vector(1);
    x[0]          = 4.0;
    auto result   = check_gradient(*a, *b, x);

    EXPECT_TRUE(result.passed);
    EXPECT_NEAR(result.max_abs_error, 0.0, 1e-14);
}

// ---------------------------------------------------------------------------
// Problem integration: set_jacobian_provider / derivatives()
// ---------------------------------------------------------------------------

TEST(LeastSquaresProblemIntegration, SetJacobianProviderPopulatesFlag)
{
    const auto&                tp = make_rosenbrock_problem();
    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;

    EXPECT_FALSE(p.has_jacobian_provider());

    auto provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);
    p.set_jacobian_provider(provider);

    EXPECT_TRUE(p.has_jacobian_provider());
    EXPECT_NE(p.derivative_provider(), nullptr);
}

TEST(LeastSquaresProblemIntegration, DerivativesFluentAlias)
{
    const auto&                tp = make_rosenbrock_problem();
    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;

    auto provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);
    p.derivatives(provider);

    EXPECT_TRUE(p.has_jacobian_provider());
}

TEST(LeastSquaresProblemIntegration, SetNullProviderClearsTheSlot)
{
    const auto&                tp = make_rosenbrock_problem();
    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;

    auto provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);
    p.set_jacobian_provider(provider);
    EXPECT_TRUE(p.has_jacobian_provider());

    p.set_jacobian_provider(nullptr);
    EXPECT_FALSE(p.has_jacobian_provider());
    EXPECT_EQ(p.derivative_provider(), nullptr);
}

TEST(LeastSquaresProblemIntegration, AutoDiffTagWithoutFactoryThrows)
{
    api::least_squares_problem p;
    p.num_parameters = 2;
    p.num_residuals  = 2;

    EXPECT_THROW(p.derivatives(api::auto_diff()), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// AutoDiffJacobianProvider (Ceres-guarded)
// ---------------------------------------------------------------------------

#if SOLVERS_HAS_CERES

TEST(AutoDiffJacobianProvider, Metadata)
{
    const auto& tp = make_rosenbrock_problem();
    auto provider  = auto_diff(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);

    EXPECT_EQ(provider->num_parameters(), tp.num_parameters);
    EXPECT_EQ(provider->num_residuals(), tp.num_residuals);
    EXPECT_EQ(provider->source(), api::derivative_mode::automatic_differentiation);
    EXPECT_NE(provider->ceres_factory(), nullptr);
}

TEST(AutoDiffJacobianProvider, ComputeResiduals)
{
    const auto& tp = make_rosenbrock_problem();
    auto provider  = auto_diff(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    matrix_type       J = make_matrix(tp.num_residuals, tp.num_parameters);
    provider->compute(x, r, J);

    vector_type r_expected = make_vector(tp.num_residuals);
    tp.residuals(x, r_expected);

    EXPECT_NEAR(r[0], r_expected[0], 1e-12);
    EXPECT_NEAR(r[1], r_expected[1], 1e-12);
}

TEST(AutoDiffJacobianProvider, JacobianMatchesAnalytic)
{
    const auto& tp = make_rosenbrock_problem();
    auto        ad_provider =
        auto_diff(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    auto analytic_provider =
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals);

    const vector_type x      = rosenbrock_x();
    auto              result = check_jacobian(*analytic_provider, *ad_provider, x, 1e-8);

    EXPECT_TRUE(result.passed) << result.summary;
}

TEST(AutoDiffJacobianProvider, ResidualsOnly)
{
    const auto& tp = make_rosenbrock_problem();
    auto provider  = auto_diff(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);

    const vector_type x = rosenbrock_x();
    vector_type       r = make_vector(tp.num_residuals);
    provider->residuals_only(x, r);

    vector_type r_expected = make_vector(tp.num_residuals);
    tp.residuals(x, r_expected);

    EXPECT_NEAR(r[0], r_expected[0], 1e-12);
    EXPECT_NEAR(r[1], r_expected[1], 1e-12);
}

TEST(AutoDiffJacobianProvider, SetJacobianProviderExposesCeresFactory)
{
    const auto& tp = make_rosenbrock_problem();

    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;

    auto provider = auto_diff(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    p.set_jacobian_provider(provider);

    // The Ceres factory is discovered from the provider, not stored separately.
    ASSERT_NE(p.derivative_provider(), nullptr);
    EXPECT_NE(p.derivative_provider()->ceres_factory(), nullptr);
}

TEST(AutoDiffJacobianProvider, LeastSquaresConvenienceFactory)
{
    auto problem = least_squares(testing::RosenbrocResiduals{}, 2u, 2u);

    EXPECT_EQ(problem.num_parameters, 2u);
    EXPECT_EQ(problem.num_residuals, 2u);
    EXPECT_TRUE(static_cast<bool>(problem.residuals));
    // The model factory is internal; its observable effect is that
    // derivatives(auto_diff()) succeeds.
    EXPECT_NO_THROW(problem.derivatives(api::auto_diff()));
}

TEST(AutoDiffJacobianProvider, AutoDiffTagInstantiatesProvider)
{
    auto problem = least_squares(testing::RosenbrocResiduals{}, 2u, 2u);

    EXPECT_FALSE(problem.has_jacobian_provider());
    problem.derivatives(api::auto_diff());
    EXPECT_TRUE(problem.has_jacobian_provider());
    ASSERT_NE(problem.derivative_provider(), nullptr);
    EXPECT_NE(problem.derivative_provider()->ceres_factory(), nullptr);
}

// ---------------------------------------------------------------------------
// End-to-end solve: all providers produce the same minimum
// ---------------------------------------------------------------------------

TEST(EndToEnd, AnalyticProviderLM)
{
    const auto&                tp = make_rosenbrock_problem();
    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;
    p.set_jacobian_provider(
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals));

    vector_type x0     = to_vector_type(tp.initial_guess);
    auto        result = api::solve(p, x0);

    ASSERT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters[0], tp.expected_solution[0], 1e-4);
    EXPECT_NEAR(result.parameters[1], tp.expected_solution[1], 1e-4);
}

TEST(EndToEnd, FiniteDifferenceProviderLM)
{
    const auto&                tp = make_rosenbrock_problem();
    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;
    p.set_jacobian_provider(finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals));

    vector_type x0     = to_vector_type(tp.initial_guess);
    auto        result = api::solve(p, x0);

    ASSERT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters[0], tp.expected_solution[0], 1e-3);
    EXPECT_NEAR(result.parameters[1], tp.expected_solution[1], 1e-3);
}

TEST(EndToEnd, AutoDiffProviderLM)
{
    auto problem = least_squares(testing::RosenbrocResiduals{}, 2u, 2u);
    problem.derivatives(api::auto_diff());

    vector_type x0 = make_vector(2);
    x0[0]          = -1.2;
    x0[1]          = 1.0;
    auto result    = api::solve(problem, x0);

    ASSERT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters[0], 1.0, 1e-4);
    EXPECT_NEAR(result.parameters[1], 1.0, 1e-4);
}

TEST(EndToEnd, AllProvidersAgreeOnSolution)
{
    const auto&       tp = make_rosenbrock_problem();
    const vector_type x0 = to_vector_type(tp.initial_guess);

    // Analytic
    api::least_squares_problem pa;
    pa.num_parameters = tp.num_parameters;
    pa.num_residuals  = tp.num_residuals;
    pa.residuals      = tp.residuals;
    pa.set_jacobian_provider(
        analytic_jacobian(tp.residuals, tp.jacobian, tp.num_parameters, tp.num_residuals));
    auto ra = api::solve(pa, x0);

    // Finite difference
    api::least_squares_problem pfd;
    pfd.num_parameters = tp.num_parameters;
    pfd.num_residuals  = tp.num_residuals;
    pfd.residuals      = tp.residuals;
    pfd.set_jacobian_provider(finite_difference(tp.residuals, tp.num_parameters, tp.num_residuals));
    auto rfd = api::solve(pfd, x0);

    // Autodiff (Ceres)
    auto pad = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    pad.derivatives(api::auto_diff());
    auto rad = api::solve(pad, to_vector_type(tp.initial_guess));

    ASSERT_TRUE(ra.converged());
    ASSERT_TRUE(rfd.converged());
    ASSERT_TRUE(rad.converged());

    EXPECT_NEAR(ra.parameters[0], rfd.parameters[0], 1e-3);
    EXPECT_NEAR(ra.parameters[1], rfd.parameters[1], 1e-3);
    EXPECT_NEAR(ra.parameters[0], rad.parameters[0], 1e-3);
    EXPECT_NEAR(ra.parameters[1], rad.parameters[1], 1e-3);
}

#endif  // SOLVERS_HAS_CERES

}  // namespace
}  // namespace solverslib
