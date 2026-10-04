#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "solvers/api/derivative_provider.h"
#include "solvers/api/detail/evaluators.h"
#include "solvers/api/roots.h"
#include "solvers/api/solve.h"

namespace solverslib::api
{
namespace
{
using detail::evaluation_status;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// -- finite-difference stencils at and across bounds --------------------------------------
TEST(ApiPaths, GradientStencilsStayInsideTheBox)
{
    api::bounds box;
    box.lower = {0.0, 0.0, 2.0};
    box.upper = {1.0, 1.0, 2.0};  // third coordinate is pinned: lower == upper
    auto objective = [&](const vector_type& x)
    {
        EXPECT_GE(x[0], 0.0);
        EXPECT_LE(x[0], 1.0);
        EXPECT_GE(x[1], 0.0);
        EXPECT_LE(x[1], 1.0);
        return 3.0 * x[0] + 2.0 * x[1] + x[2];
    };
    detail::finite_difference_gradient_evaluator fd(3, objective, 1e-3, box);
    for (const auto& corner : {std::pair{0.0, 1.0}, std::pair{1.0, 0.0}})
    {
        vector_type x(3);
        x << corner.first, corner.second, 2.0;
        double      value = 0.0;
        vector_type g(3);
        ASSERT_EQ(fd.evaluate(x, value, &g), evaluation_status::ok);
        EXPECT_NEAR(g[0], 3.0, 1e-8);
        EXPECT_NEAR(g[1], 2.0, 1e-8);
        EXPECT_DOUBLE_EQ(g[2], 0.0);  // no room to step
        vector_type g2(3);
        ASSERT_EQ(fd.gradient(x, g2), evaluation_status::ok);
        EXPECT_NEAR((g - g2).norm(), 0.0, 1e-12);
    }
    EXPECT_GT(fd.counters().objective_evaluations, 0u);
    EXPECT_EQ(fd.source(), derivative_mode::finite_difference);
    EXPECT_EQ(fd.num_parameters(), 3u);
}

TEST(ApiPaths, JacobianStencilWithNoRoomIsZero)
{
    api::bounds box;
    box.lower = {1.0};
    box.upper = {1.0};
    detail::finite_difference_residual_evaluator fd(
        1, 1, [](const vector_type& x, vector_type& r) { r.resize(1); r[0] = 5.0 * x[0]; }, 1e-3, box);
    vector_type x = vector_type::Constant(1, 1.0), r(1);
    matrix_type j(1, 1);
    ASSERT_EQ(fd.evaluate(x, r, &j), evaluation_status::ok);
    EXPECT_DOUBLE_EQ(j(0, 0), 0.0);
}

TEST(ApiPaths, FiniteDifferenceEvaluatorsFlagNonFiniteOutput)
{
    detail::finite_difference_gradient_evaluator grad(
        1, [](const vector_type&) { return kNaN; });
    double      v = 0.0;
    vector_type g(1);
    EXPECT_EQ(grad.evaluate(vector_type::Zero(1), v, &g), evaluation_status::invalid_trial);
    EXPECT_EQ(grad.gradient(vector_type::Zero(1), g), evaluation_status::invalid_trial);

    detail::finite_difference_residual_evaluator res(
        1, 1, [](const vector_type&, vector_type& r) { r.resize(1); r[0] = kNaN; });
    vector_type r(1);
    matrix_type j(1, 1);
    EXPECT_EQ(res.evaluate(vector_type::Zero(1), r, &j), evaluation_status::invalid_trial);
    r[0] = 1.0;
    EXPECT_EQ(res.jacobian(vector_type::Zero(1), r, j), evaluation_status::invalid_trial);
}

// -- callback evaluators without the callback they are asked for -------------------------
TEST(ApiPaths, CallbackEvaluatorsReportMissingAndBadDerivatives)
{
    detail::callback_gradient_evaluator no_grad(
        1, [](const vector_type&) { return 1.0; }, gradient_function{}, derivative_mode::supplied);
    double      v = 0.0;
    vector_type g(1);
    EXPECT_EQ(no_grad.evaluate(vector_type::Zero(1), v, &g), evaluation_status::fatal_error);
    EXPECT_EQ(no_grad.gradient(vector_type::Zero(1), g), evaluation_status::fatal_error);
    EXPECT_TRUE(no_grad.last_error().has_value());

    detail::callback_gradient_evaluator bad_grad(
        1,
        [](const vector_type&) { return 1.0; },
        [](const vector_type&, vector_type& out) { out.resize(1); out[0] = kNaN; },
        derivative_mode::supplied);
    EXPECT_EQ(bad_grad.evaluate(vector_type::Zero(1), v, &g), evaluation_status::invalid_trial);
    EXPECT_EQ(bad_grad.gradient(vector_type::Zero(1), g), evaluation_status::invalid_trial);
    EXPECT_EQ(bad_grad.source(), derivative_mode::supplied);

    detail::callback_residual_evaluator no_jac(
        1, 1, [](const vector_type&, vector_type& r) { r[0] = 1.0; });
    vector_type r(1);
    matrix_type j(1, 1);
    EXPECT_EQ(no_jac.evaluate(vector_type::Zero(1), r, &j), evaluation_status::fatal_error);
    EXPECT_EQ(no_jac.jacobian(vector_type::Zero(1), r, j), evaluation_status::fatal_error);

    detail::callback_residual_evaluator bad_jac(
        1,
        1,
        [](const vector_type&, vector_type& out) { out[0] = 1.0; },
        jacobian_function([](const vector_type&, matrix_type& out) { out(0, 0) = kNaN; }));
    EXPECT_EQ(bad_jac.evaluate(vector_type::Zero(1), r, &j), evaluation_status::invalid_trial);
    EXPECT_EQ(bad_jac.jacobian(vector_type::Zero(1), r, j), evaluation_status::invalid_trial);
}

TEST(ApiPaths, DelegatingEvaluatorUsesTheProviderForDerivativesOnly)
{
    int  residual_calls = 0;
    auto residuals      = [&](const vector_type& x, vector_type& r)
    {
        ++residual_calls;
        r.resize(1);
        r[0] = x[0] * x[0];
    };
    auto provider = analytic_jacobian(
        residuals, [](const vector_type& x, matrix_type& j) { j(0, 0) = 2.0 * x[0]; }, 1, 1);
    detail::delegating_residual_evaluator evaluator(
        residuals, detail::make_provider_evaluator(provider), derivative_mode::supplied);

    vector_type x = vector_type::Constant(1, 3.0), r(1);
    matrix_type j(1, 1);
    ASSERT_EQ(evaluator.evaluate(x, r), evaluation_status::ok);
    EXPECT_DOUBLE_EQ(r[0], 9.0);
    ASSERT_EQ(evaluator.evaluate(x, r, &j), evaluation_status::ok);
    EXPECT_DOUBLE_EQ(j(0, 0), 6.0);
    j(0, 0) = 0.0;
    ASSERT_EQ(evaluator.jacobian(x, r, j), evaluation_status::ok);
    EXPECT_DOUBLE_EQ(j(0, 0), 6.0);
    EXPECT_EQ(evaluator.metadata().source, derivative_mode::supplied);
    EXPECT_FALSE(evaluator.last_error().has_value());
    EXPECT_EQ(evaluator.counters().jacobian_evaluations, 2u);
}

TEST(ApiPaths, DelegatingEvaluatorSurfacesProviderErrors)
{
    auto residuals = [](const vector_type&, vector_type& r) { r.resize(1); r[0] = 1.0; };
    class throwing final : public JacobianProvider
    {
    public:
        void compute(const vector_type&, vector_type&, matrix_type&) const override
        {
            throw std::runtime_error("provider failed");
        }
        std::size_t     num_parameters() const override { return 1; }
        std::size_t     num_residuals() const override { return 1; }
        derivative_mode source() const override { return derivative_mode::supplied; }
    };
    detail::delegating_residual_evaluator evaluator(
        residuals, detail::make_provider_evaluator(std::make_shared<throwing>()), derivative_mode::supplied);
    vector_type x = vector_type::Zero(1), r(1);
    matrix_type j(1, 1);
    EXPECT_EQ(evaluator.evaluate(x, r, &j), evaluation_status::fatal_error);
    EXPECT_EQ(evaluator.jacobian(x, r, j), evaluation_status::fatal_error);
    EXPECT_NE(evaluator.last_error().value_or("").find("provider failed"), std::string::npos);
}

// -- problem-level errors through api::solve ---------------------------------------------------
least_squares_problem simple_problem()
{
    least_squares_problem p;
    p.num_parameters = 1;
    p.num_residuals  = 1;
    p.residuals      = [](const vector_type& x, vector_type& r) { r.resize(1); r[0] = x[0] - 1.0; };
    return p;
}

TEST(ApiPaths, OptionAndPolicyErrorsAreStatuses)
{
    const vector_type x0 = vector_type::Zero(1);
    solve_options     bad;
    bad.function_tolerance = -1.0;
    EXPECT_EQ(solve(simple_problem(), x0, bad).status, solver_status::invalid_problem);
    EXPECT_EQ(solve(simple_problem(), vector_type::Constant(1, kNaN)).status,
        solver_status::invalid_problem);

    solve_options supplied;
    supplied.derivatives = derivative_mode::supplied;
    EXPECT_EQ(solve(simple_problem(), x0, supplied).status, solver_status::invalid_problem);

    solve_options ad;
    ad.derivatives = derivative_mode::automatic_differentiation;
    EXPECT_EQ(solve(simple_problem(), x0, ad).status, solver_status::unsupported_capability);

    // Bounds with the wrong size are rejected before any callback runs.
    auto bounded         = simple_problem();
    bounded.bounds.lower = {0.0, 1.0};
    EXPECT_EQ(solve(bounded, x0).status, solver_status::invalid_problem);

    // Pinning a combination no route implements.
    solve_options impossible;
    impossible.backend   = backend::ipopt;
    impossible.algorithm = algorithm::levenberg_marquardt;
    const auto result    = solve(simple_problem(), x0, impossible);
    EXPECT_EQ(result.status, solver_status::unsupported_capability);
    EXPECT_FALSE(result.message.empty());
}

TEST(ApiPaths, ObjectiveDerivativePolicyErrors)
{
    optimization_problem p;
    p.num_parameters = 1;
    p.objective      = [](const vector_type& x) { return x[0] * x[0]; };
    const vector_type x0 = vector_type::Constant(1, 1.0);

    solve_options supplied;
    supplied.derivatives = derivative_mode::supplied;
    EXPECT_EQ(solve(p, x0, supplied).status, solver_status::invalid_problem);

    solve_options ad;
    ad.derivatives = derivative_mode::automatic_differentiation;
    EXPECT_EQ(solve(p, x0, ad).status, solver_status::unsupported_capability);

    p.constraints.num_equality = 1;
    solve_options native;
    native.backend = backend::native;
    EXPECT_EQ(solve(p, x0, native).status, solver_status::unsupported_capability);
}

TEST(ApiPaths, ToStringCoversEveryEnumerator)
{
    for (const auto s : {solver_status::converged, solver_status::max_iterations,
             solver_status::invalid_problem, solver_status::numerical_failure,
             solver_status::infeasible, solver_status::unsupported_capability,
             solver_status::backend_unavailable, solver_status::user_stopped,
             solver_status::stalled})
    {
        EXPECT_STRNE(to_string(s), "unknown");
    }
    for (const auto a : {algorithm::automatic, algorithm::levenberg_marquardt,
             algorithm::gauss_newton, algorithm::lbfgs, algorithm::pounders,
             algorithm::interior_point, algorithm::newton_krylov,
             algorithm::riemann_normal_coordinate_lm})
    {
        EXPECT_STRNE(to_string(a), "unknown");
    }
    for (const auto b : {backend::automatic, backend::native, backend::ipopt, backend::petsc_tao,
             backend::pounders, backend::ceres})
    {
        EXPECT_STRNE(to_string(b), "unknown");
    }
    for (const auto d : {derivative_mode::automatic, derivative_mode::supplied,
             derivative_mode::automatic_differentiation, derivative_mode::finite_difference})
    {
        EXPECT_STRNE(to_string(d), "unknown");
    }
    for (const auto m : {root_method::bisection, root_method::false_position, root_method::ridders,
             root_method::brent, root_method::dekker, root_method::secant,
             root_method::newton_raphson})
    {
        EXPECT_STRNE(to_string(m), "unknown");
    }
}

// -- root-finding argument and failure paths ------------------------------------------------------
TEST(ApiPaths, FindRootRejectsBadArgumentsInEveryOverload)
{
    const scalar_function          f  = [](double x) { return x - 1.0; };
    const scalar_function_gradient fd = [](double x, double& d) { d = 1.0; return x - 1.0; };

    root_options bad_tol;
    bad_tol.tolerance_function = -1.0;
    EXPECT_EQ(find_root(root_method::brent, f, 0.0, 2.0, bad_tol).status,
        solver_status::invalid_problem);
    root_options bad_target;
    bad_target.target = kNaN;
    EXPECT_EQ(find_root(root_method::dekker, fd, 0.0, 2.0, bad_target).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::newton_raphson, fd, 0.0, bad_target).status,
        solver_status::invalid_problem);

    EXPECT_EQ(find_root(root_method::brent, scalar_function{}, 0.0, 2.0).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::dekker, scalar_function_gradient{}, 0.0, 2.0).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::newton_raphson, fd, kNaN).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::newton_raphson, scalar_function_gradient{}, 0.0).status,
        solver_status::invalid_problem);
}

TEST(ApiPaths, DekkerAndNewtonFailuresAreStatuses)
{
    // No sign change.
    const auto flat = find_root(root_method::dekker,
        scalar_function_gradient([](double x, double& d) { d = 2.0 * x; return x * x + 1.0; }),
        0.0,
        2.0);
    EXPECT_EQ(flat.status, solver_status::invalid_problem);
    EXPECT_EQ(flat.evaluations, 2u);

    // Non-finite at an endpoint of the bracket check.
    const auto nan_end = find_root(root_method::dekker,
        scalar_function_gradient([](double x, double& d) { d = 1.0; return x > 1.5 ? kNaN : x - 1.0; }),
        0.0,
        2.0);
    EXPECT_EQ(nan_end.status, solver_status::numerical_failure);

    const auto nan_value = find_root(root_method::brent,
        scalar_function([](double x) { return x > 1.5 ? kNaN : x - 1.0; }),
        0.0,
        2.0);
    EXPECT_EQ(nan_value.status, solver_status::numerical_failure);

    const auto thrown = find_root(root_method::brent,
        scalar_function([](double) -> double { throw std::runtime_error("bad endpoint"); }),
        0.0,
        2.0);
    EXPECT_EQ(thrown.status, solver_status::numerical_failure);

    // Newton with a non-finite derivative.
    const auto nan_slope = find_root(root_method::newton_raphson,
        scalar_function_gradient([](double x, double& d) { d = kNaN; return x - 1.0; }),
        3.0);
    EXPECT_EQ(nan_slope.status, solver_status::numerical_failure);

    // A converged Newton run evaluates the residual at its answer.
    const auto ok = find_root(root_method::newton_raphson,
        scalar_function_gradient([](double x, double& d) { d = 2.0 * x; return x * x - 4.0; }),
        3.0);
    EXPECT_TRUE(ok.converged());
    EXPECT_NEAR(ok.root, 2.0, 1e-8);
}
}  // namespace
}  // namespace solverslib::api
