#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "detail/support.h"
#include "optimization_test_problems.h"
#include "solver_options/solver_options_lm.h"
#include "solvers/api/derivative_provider.h"
#include "solvers/api/detail/evaluators.h"
#include "solvers/api/solve.h"
#include "solvers/ceres_solver.h"
#include "solvers/levenberg_marquardt_solver.h"

#if SOLVERS_HAS_CERES
#include "solvers/integrations/autodiff_provider.h"
#endif

namespace solverslib::api
{
namespace
{
using detail::evaluation_status;

// Rosenbrock residual form with call counters on the user's callbacks, so the
// evaluator's own counters can be checked against ground truth.
struct counted_problem
{
    std::size_t residual_calls = 0;
    std::size_t jacobian_calls = 0;

    least_squares_problem make(bool with_jacobian)
    {
        const auto            tp = testing::make_rosenbrock_problem();
        least_squares_problem problem;
        problem.num_parameters = tp.num_parameters;
        problem.num_residuals  = tp.num_residuals;
        problem.residuals      = [this, f = tp.residuals](const vector_type& x, vector_type& r)
        {
            ++residual_calls;
            f(x, r);
        };
        if (with_jacobian)
        {
            problem.set_jacobian(
                [this, j = tp.jacobian](const vector_type& x, matrix_type& jac)
                {
                    ++jacobian_calls;
                    j(x, jac);
                });
        }
        return problem;
    }
};

const vector_type kStart = (vector_type(2) << -1.2, 1.0).finished();

// -- exact evaluation counts ----------------------------------------------------
// With geodesic acceleration off, LM evaluates residuals once per iteration plus
// once at the start; Jacobians once at the start plus once per accepted step.
TEST(Evaluators, KernelEvaluatesResidualsOncePerTrial)
{
    counted_problem counted;
    const auto      problem = counted.make(true);

    detail::delegating_residual_evaluator evaluator(problem.residuals,
        detail::make_provider_evaluator(problem.derivative_provider()),
        derivative_mode::supplied);
    levenberg_marquardt_solver            solver(evaluator);
    auto options = solver_options_lm_builder().with_geodesic_acceleration(false).build();

    vector_type x      = kStart;
    const auto  result = solver.solve(x, *options);
    ASSERT_TRUE(result.converged());

    EXPECT_EQ(counted.residual_calls, 1 + result.iterations);
    EXPECT_EQ(evaluator.counters().residual_evaluations, counted.residual_calls);
    EXPECT_EQ(evaluator.counters().jacobian_evaluations, counted.jacobian_calls);
    EXPECT_GE(counted.jacobian_calls, 1u);
    EXPECT_LE(counted.jacobian_calls, 1 + result.iterations);
}

TEST(Evaluators, KernelCountsFiniteDifferenceStencils)
{
    counted_problem counted;
    const auto      problem = counted.make(false);

    detail::finite_difference_residual_evaluator evaluator(
        problem.num_parameters, problem.num_residuals, problem.residuals);
    levenberg_marquardt_solver solver(evaluator);
    auto options = solver_options_lm_builder().with_geodesic_acceleration(false).build();

    vector_type x      = kStart;
    const auto  result = solver.solve(x, *options);
    ASSERT_TRUE(result.converged());

    // 1 initial + 1 per iteration + a 2n central stencil per Jacobian.
    const std::size_t expected =
        1 + result.iterations +
        2 * problem.num_parameters * evaluator.counters().jacobian_evaluations;
    EXPECT_EQ(counted.residual_calls, expected);
    EXPECT_EQ(evaluator.counters().residual_evaluations, counted.residual_calls);
}

// Through api::solve the reported counters must equal what the user's own
// callbacks observed, whatever the kernel does internally.
TEST(Evaluators, ApiCountersMatchCallbackCalls)
{
    for (const bool with_jacobian : {true, false})
    {
        counted_problem counted;
        const auto      problem = counted.make(with_jacobian);

        solve_options options;
        options.backend   = backend::native;
        const auto result = solve(problem, kStart, options);
        ASSERT_TRUE(result.converged()) << result.message;
        EXPECT_EQ(result.residual_evaluations, counted.residual_calls);
        EXPECT_EQ(result.jacobian_evaluations,
            with_jacobian ? counted.jacobian_calls : result.jacobian_evaluations);
        ASSERT_TRUE(result.effective_derivative_source.has_value());
        EXPECT_EQ(*result.effective_derivative_source,
            with_jacobian ? derivative_mode::supplied : derivative_mode::finite_difference);
    }
}

// -- evaluator unit behavior ------------------------------------------------------
TEST(Evaluators, FiniteDifferenceJacobianMatchesAnalytic)
{
    const auto                                   tp = testing::make_rosenbrock_problem();
    detail::finite_difference_residual_evaluator fd(2, 2, tp.residuals);

    const vector_type x = (vector_type(2) << 0.7, -0.3).finished();
    vector_type       r(2);
    matrix_type       j_fd(2, 2), j_exact(2, 2);
    ASSERT_EQ(fd.evaluate(x, r, &j_fd), evaluation_status::ok);
    tp.jacobian(x, j_exact);
    EXPECT_LT((j_fd - j_exact).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_EQ(fd.counters().jacobian_evaluations, 1u);
    EXPECT_EQ(fd.counters().residual_evaluations, 1u + 2u * 2u);
}

TEST(Evaluators, FiniteDifferenceStencilNeverLeavesBounds)
{
    api::bounds box;
    box.lower = {0.0, 0.0};
    box.upper = {1.0, 1.0};

    residual_function residual = [&](const vector_type& x, vector_type& r)
    {
        ASSERT_GE(x[0], box.lower[0]);
        ASSERT_LE(x[0], box.upper[0]);
        ASSERT_GE(x[1], box.lower[1]);
        ASSERT_LE(x[1], box.upper[1]);
        r.resize(1);
        r[0] = 3.0 * x[0] + 2.0 * x[1];
    };
    detail::finite_difference_residual_evaluator fd(2, 1, residual, 1e-3, box);

    for (const auto& corner : {std::pair{0.0, 0.0}, std::pair{1.0, 1.0}, std::pair{0.0, 1.0}})
    {
        const vector_type x = (vector_type(2) << corner.first, corner.second).finished();
        vector_type       r(1);
        matrix_type       j(1, 2);
        ASSERT_EQ(fd.evaluate(x, r, &j), evaluation_status::ok);
        EXPECT_NEAR(j(0, 0), 3.0, 1e-8);
        EXPECT_NEAR(j(0, 1), 2.0, 1e-8);
    }
}

TEST(Evaluators, CallbackEvaluatorClassifiesFailures)
{
    bool                                throw_now = false;
    double                              value     = 1.0;
    detail::callback_residual_evaluator evaluator(1,
        1,
        [&](const vector_type&, vector_type& r)
        {
            if (throw_now)
            {
                throw std::runtime_error("boom");
            }
            r[0] = value;
        });
    const vector_type                   x = vector_type::Zero(1);
    vector_type                         r(1);

    EXPECT_EQ(evaluator.evaluate(x, r), evaluation_status::ok);
    value = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(evaluator.evaluate(x, r), evaluation_status::invalid_trial);
    throw_now = true;
    EXPECT_EQ(evaluator.evaluate(x, r), evaluation_status::fatal_error);
    EXPECT_TRUE(evaluator.last_error().value_or("") == "boom");
    EXPECT_EQ(evaluator.counters().residual_evaluations, 3u);
}

// -- failure handling inside a solve ------------------------------------------------
TEST(Evaluators, NonFiniteTrialIsRejectedNotFatal)
{
    // r(x) = x - 3, but NaN for x > 5: LM's first big step may land there.
    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 1;
    problem.residuals      = [](const vector_type& x, vector_type& r)
    { r[0] = x[0] > 5.0 ? std::numeric_limits<double>::quiet_NaN() : x[0] - 3.0; };
    problem.set_jacobian([](const vector_type&, matrix_type& j) { j(0, 0) = 1.0; });

    solve_options options;
    options.backend   = backend::native;
    const auto result = solve(problem, vector_type::Constant(1, 0.0), options);
    EXPECT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters[0], 3.0, 1e-6);
}

TEST(Evaluators, ThrowingCallbackBecomesNumericalFailureNotException)
{
    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 1;
    int calls              = 0;
    problem.residuals      = [&](const vector_type& x, vector_type& r)
    {
        if (++calls > 2)
        {
            throw std::runtime_error("model blew up");
        }
        r[0] = x[0] - 3.0;
    };
    problem.set_jacobian([](const vector_type&, matrix_type& j) { j(0, 0) = 1.0; });

    solve_options options;
    options.backend             = backend::native;
    options.function_tolerance  = 0.0;
    options.parameter_tolerance = 0.0;
    options.max_iterations      = 50;
    solver_result result;
    ASSERT_NO_THROW(result = solve(problem, vector_type::Constant(1, 0.0), options));
    EXPECT_EQ(result.status, solver_status::numerical_failure);
    EXPECT_NE(result.message.find("model blew up"), std::string::npos) << result.message;
}

TEST(Evaluators, NonFiniteInitialPointFailsBeforeIterating)
{
    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 1;
    problem.residuals      = [](const vector_type&, vector_type& r)
    { r[0] = std::numeric_limits<double>::quiet_NaN(); };
    solve_options options;
    options.backend   = backend::native;
    const auto result = solve(problem, vector_type::Zero(1), options);
    EXPECT_EQ(result.status, solver_status::numerical_failure);
    EXPECT_EQ(result.iterations, 0u);
}

// -- derivative-policy matrix ----------------------------------------------------------
enum class source_kind
{
    none,
    callback,
    analytic_provider,
    fd_provider,
    ad_provider
};

least_squares_problem make_with(source_kind kind)
{
    const auto            tp = testing::make_rosenbrock_problem();
    least_squares_problem problem;
    problem.num_parameters = tp.num_parameters;
    problem.num_residuals  = tp.num_residuals;
    problem.residuals      = tp.residuals;
    switch (kind)
    {
    case source_kind::none:
        break;
    case source_kind::callback:
        problem.set_jacobian(tp.jacobian);
        break;
    case source_kind::analytic_provider:
        problem.set_jacobian_provider(analytic_jacobian(tp.residuals, tp.jacobian, 2, 2));
        break;
    case source_kind::fd_provider:
        problem.set_jacobian_provider(finite_difference(tp.residuals, 2, 2));
        break;
    case source_kind::ad_provider:
#if SOLVERS_HAS_CERES
        problem = solverslib::least_squares(testing::RosenbrocResiduals{}, 2, 2);
        problem.derivatives(solverslib::auto_diff());
#endif
        break;
    }
    return problem;
}

struct policy_case
{
    source_kind     kind;
    derivative_mode policy;
    // Expected: a concrete source when it should solve, otherwise the status.
    std::optional<derivative_mode> source;
    solver_status                  failure;
};

TEST(Evaluators, DerivativePolicyMatrix)
{
    using S        = source_kind;
    using D        = derivative_mode;
    const auto ok  = solver_status::converged;
    const auto bad = solver_status::invalid_problem;
    const auto uns = solver_status::unsupported_capability;

    std::vector<policy_case> cases = {
        {S::none, D::automatic, D::finite_difference, ok},
        {S::none, D::supplied, std::nullopt, bad},
        {S::none, D::automatic_differentiation, std::nullopt, uns},
        {S::none, D::finite_difference, D::finite_difference, ok},
        {S::callback, D::automatic, D::supplied, ok},
        {S::callback, D::supplied, D::supplied, ok},
        {S::callback, D::automatic_differentiation, std::nullopt, uns},
        {S::callback, D::finite_difference, D::finite_difference, ok},
        {S::analytic_provider, D::automatic, D::supplied, ok},
        {S::analytic_provider, D::supplied, D::supplied, ok},
        {S::analytic_provider, D::automatic_differentiation, std::nullopt, uns},
        {S::analytic_provider, D::finite_difference, D::finite_difference, ok},
        {S::fd_provider, D::automatic, D::finite_difference, ok},
        {S::fd_provider, D::supplied, std::nullopt, bad},
        {S::fd_provider, D::automatic_differentiation, std::nullopt, uns},
        {S::fd_provider, D::finite_difference, D::finite_difference, ok},
    };
#if SOLVERS_HAS_CERES
    cases.push_back({S::ad_provider, D::automatic, D::automatic_differentiation, ok});
    cases.push_back({S::ad_provider, D::supplied, std::nullopt, bad});
    cases.push_back(
        {S::ad_provider, D::automatic_differentiation, D::automatic_differentiation, ok});
    cases.push_back({S::ad_provider, D::finite_difference, D::finite_difference, ok});
#endif

    std::vector<backend> backends = {backend::native};
    if (ceres_solver::is_supported())
    {
        backends.push_back(backend::ceres);
    }

    for (const backend chosen : backends)
    {
        for (const auto& c : cases)
        {
            SCOPED_TRACE(std::string(to_string(chosen)) +
                         " kind=" + std::to_string(static_cast<int>(c.kind)) +
                         " policy=" + to_string(c.policy));
            solve_options options;
            options.backend     = chosen;
            options.derivatives = c.policy;
            const auto result   = solve(make_with(c.kind), kStart, options);
            if (c.source)
            {
                EXPECT_TRUE(result.converged()) << result.message;
                ASSERT_TRUE(result.effective_derivative_source.has_value());
                EXPECT_EQ(*result.effective_derivative_source, *c.source);
            }
            else
            {
                EXPECT_EQ(result.status, c.failure) << result.message;
            }
        }
    }
}

// -- objective side ----------------------------------------------------------------------
optimization_problem make_quadratic(bool with_gradient)
{
    optimization_problem problem;
    problem.num_parameters = 2;
    problem.objective      = [](const vector_type& x)
    { return (x - vector_type::Constant(2, 2.0)).squaredNorm(); };
    if (with_gradient)
    {
        problem.set_gradient([](const vector_type& x, vector_type& g)
            { g = 2.0 * (x - vector_type::Constant(2, 2.0)); });
    }
    return problem;
}

TEST(Evaluators, ObjectiveFiniteDifferenceGradientIsExplicitOptIn)
{
    // Automatic still refuses to guess a gradient...
    EXPECT_EQ(solve(make_quadratic(false), vector_type::Zero(2)).status,
        solver_status::unsupported_capability);

    // ...but an explicit request differentiates the objective numerically.
    solve_options options;
    options.derivatives = derivative_mode::finite_difference;
    const auto result   = solve(make_quadratic(false), vector_type::Zero(2), options);
    EXPECT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters[0], 2.0, 1e-4);
    ASSERT_TRUE(result.effective_derivative_source.has_value());
    EXPECT_EQ(*result.effective_derivative_source, derivative_mode::finite_difference);
}

TEST(Evaluators, ObjectivePolicyIsHonoredOnTheNativePath)
{
    auto problem  = make_quadratic(true);
    auto gradient = [](const vector_type& x, vector_type& g)
    { g = 2.0 * (x - vector_type::Constant(2, 2.0)); };
    problem.set_gradient_provider(analytic_gradient(gradient, 2));

    solve_options options;
    auto          result = solve(problem, vector_type::Zero(2), options);
    ASSERT_TRUE(result.effective_derivative_source.has_value());
    EXPECT_EQ(*result.effective_derivative_source, derivative_mode::supplied);

    options.derivatives = derivative_mode::automatic_differentiation;
    EXPECT_EQ(solve(problem, vector_type::Zero(2), options).status,
        solver_status::unsupported_capability);

    options.derivatives = derivative_mode::supplied;
    EXPECT_TRUE(solve(problem, vector_type::Zero(2), options).converged());
}
}  // namespace
}  // namespace solverslib::api
