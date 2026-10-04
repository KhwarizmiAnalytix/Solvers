#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "solvers/api/roots.h"
#include "solvers/polynomial_solver.h"
#include "solvers/root_finding_algorithms.h"

namespace solverslib::api
{
namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// One call shape per method: value-only bracket, derivative bracket (dekker),
// two starting points (secant) or one (newton_raphson).
struct method_case
{
    root_method method;
    bool        bracketing;  // requires a sign change on [a, b]
};

std::vector<method_case> all_methods()
{
    return {{root_method::bisection, true},
        {root_method::false_position, true},
        {root_method::ridders, true},
        {root_method::brent, true},
        {root_method::dekker, true},
        {root_method::secant, false},
        {root_method::newton_raphson, false}};
}

// Solves  g(x) = target  for a smooth g given with its derivative, using whichever
// overload the method needs. For newton_raphson `a` is the starting point.
root_result solve_with(root_method       method,
    const std::function<double(double)>& g,
    const std::function<double(double)>& dg,
    double                               a,
    double                               b,
    const root_options&                  options)
{
    const scalar_function          value_only      = g;
    const scalar_function_gradient with_derivative = [g, dg](double x, double& d)
    {
        d = dg(x);
        return g(x);
    };
    switch (method)
    {
    case root_method::dekker:
        return find_root(method, with_derivative, a, b, options);
    case root_method::newton_raphson:
        return find_root(method, with_derivative, a, options);
    default:
        return find_root(method, value_only, a, b, options);
    }
}

// The default budget (50 iterations, machine-epsilon tolerances) is tight or
// unreachable for the slow bracketing methods on some problems; the contract
// tests that expect convergence give them room.
root_options generous()
{
    root_options options;
    options.max_iterations      = 500;
    options.tolerance_function  = 1e-12;
    options.tolerance_parameter = 1e-12;
    return options;
}

std::string name_of(root_method method)
{
    return to_string(method);
}

// -- shared contract across every method -------------------------------------------------
TEST(FindRoot, ConvergesOnASmoothProblem)
{
    for (const auto& c : all_methods())
    {
        SCOPED_TRACE(name_of(c.method));
        const auto result = solve_with(
            c.method,
            [](double x) { return x * x - 2.0; },
            [](double x) { return 2.0 * x; },
            c.bracketing ? 0.0 : 1.0,
            2.0,
            generous());
        EXPECT_TRUE(result.converged()) << result.message;
        EXPECT_NEAR(result.root, std::sqrt(2.0), 1e-9);
        EXPECT_LT(std::abs(result.residual), 1e-8);
        EXPECT_GT(result.evaluations, 0u);
        EXPECT_LE(result.iterations, generous().max_iterations);
    }
}

TEST(FindRoot, TargetOffsetShiftsTheEquation)
{
    for (const auto& c : all_methods())
    {
        SCOPED_TRACE(name_of(c.method));
        root_options options = generous();
        options.target       = 8.0;  // solve x^3 = 8
        const auto result    = solve_with(
            c.method,
            [](double x) { return x * x * x; },
            [](double x) { return 3.0 * x * x; },
            c.bracketing ? 0.0 : 1.5,
            5.0,
            options);
        EXPECT_TRUE(result.converged()) << result.message;
        EXPECT_NEAR(result.root, 2.0, 1e-8);
        // The residual is measured against the target, not against zero.
        EXPECT_LT(std::abs(result.residual), 1e-6);
    }
}

TEST(FindRoot, RootAtAnEndpointOrStartIsReturnedExactly)
{
    for (const auto& c : all_methods())
    {
        SCOPED_TRACE(name_of(c.method));
        const auto f  = [](double x) { return x - 1.0; };
        const auto df = [](double) { return 1.0; };
        if (c.bracketing)
        {
            const auto at_a = solve_with(c.method, f, df, 1.0, 3.0, {});
            ASSERT_TRUE(at_a.converged()) << at_a.message;
            EXPECT_DOUBLE_EQ(at_a.root, 1.0);
            const auto at_b = solve_with(c.method, f, df, -1.0, 1.0, {});
            ASSERT_TRUE(at_b.converged()) << at_b.message;
            EXPECT_DOUBLE_EQ(at_b.root, 1.0);
        }
        else
        {
            const auto result = solve_with(c.method, f, df, 1.0, 2.0, {});
            ASSERT_TRUE(result.converged()) << result.message;
            EXPECT_NEAR(result.root, 1.0, 1e-12);
        }
    }
}

TEST(FindRoot, InvalidBracketIsReportedWithoutSearching)
{
    for (const auto& c : all_methods())
    {
        if (!c.bracketing)
        {
            continue;
        }
        SCOPED_TRACE(name_of(c.method));
        const auto result = solve_with(
            c.method,
            [](double x) { return x * x + 1.0; },  // no real root
            [](double x) { return 2.0 * x; },
            0.0,
            2.0,
            {});
        EXPECT_EQ(result.status, solver_status::invalid_problem);
        EXPECT_NE(result.message.find("bracket"), std::string::npos) << result.message;
        EXPECT_EQ(result.evaluations, 2u) << "only the two bracket checks should have run";
        EXPECT_FALSE(result.has_usable_iterate());
    }
}

TEST(FindRoot, IterationLimitKeepsTheBestEstimate)
{
    for (const auto& c : all_methods())
    {
        SCOPED_TRACE(name_of(c.method));
        root_options options;
        options.max_iterations      = 2;
        options.tolerance_function  = 0.0;  // never satisfied
        options.tolerance_parameter = 0.0;
        const auto result           = solve_with(
            c.method,
            [](double x) { return x * x * x - 2.0; },
            [](double x) { return 3.0 * x * x; },
            c.bracketing ? 0.0 : 3.0,
            c.bracketing ? 3.0 : 2.5,
            options);
        EXPECT_EQ(result.status, solver_status::max_iterations) << result.message;
        EXPECT_TRUE(result.has_usable_iterate());
        EXPECT_FALSE(result.converged());
        EXPECT_TRUE(std::isfinite(result.root));
        EXPECT_TRUE(std::isfinite(result.residual));
        EXPECT_EQ(result.iterations, 2u);
    }
}

TEST(FindRoot, NonFiniteEvaluationIsANumericalFailure)
{
    for (const auto& c : all_methods())
    {
        SCOPED_TRACE(name_of(c.method));
        const auto result = solve_with(
            c.method,
            // Defined at the endpoints and starting points, NaN around the root.
            [](double x) { return std::abs(x - 1.2345) < 0.5 ? kNaN : x - 1.2345; },
            [](double) { return 1.0; },
            c.bracketing ? 0.0 : 0.0,
            2.0,
            {});
        EXPECT_EQ(result.status, solver_status::numerical_failure) << result.message;
        EXPECT_NE(result.message.find("non-finite"), std::string::npos) << result.message;
        EXPECT_FALSE(result.has_usable_iterate());
        EXPECT_TRUE(std::isnan(result.root));
    }
}

TEST(FindRoot, ThrowingFunctionAndVanishingDerivativeDoNotEscape)
{
    root_result thrown;
    ASSERT_NO_THROW(thrown = find_root(root_method::brent,
                        scalar_function(
                            [](double x) -> double
                            {
                                if (x > 0.2 && x < 1.9)
                                {
                                    throw std::runtime_error("model blew up");
                                }
                                return x - 1.0;
                            }),
                        0.0,
                        2.0));
    EXPECT_EQ(thrown.status, solver_status::numerical_failure);
    EXPECT_NE(thrown.message.find("model blew up"), std::string::npos) << thrown.message;

    // Newton on x^2 - 1 started at 0: the derivative vanishes immediately.
    root_result flat;
    ASSERT_NO_THROW(flat = find_root(root_method::newton_raphson,
                        scalar_function_gradient(
                            [](double x, double& d)
                            {
                                d = 2.0 * x;
                                return x * x - 1.0;
                            }),
                        0.0));
    EXPECT_EQ(flat.status, solver_status::numerical_failure);
}

TEST(FindRoot, WrongCallShapeIsRejected)
{
    const scalar_function          value_only      = [](double x) { return x - 1.0; };
    const scalar_function_gradient with_derivative = [](double x, double& d)
    {
        d = 1.0;
        return x - 1.0;
    };
    EXPECT_EQ(find_root(root_method::dekker, value_only, 0.0, 2.0).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::newton_raphson, value_only, 0.0, 2.0).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::brent, with_derivative, 0.0, 2.0).status,
        solver_status::invalid_problem);
    EXPECT_EQ(
        find_root(root_method::brent, with_derivative, 0.0).status, solver_status::invalid_problem);

    root_options bad;
    bad.max_iterations = 0;
    EXPECT_EQ(find_root(root_method::brent, value_only, 0.0, 2.0, bad).status,
        solver_status::invalid_problem);
    EXPECT_EQ(find_root(root_method::brent, value_only, 0.0, kNaN).status,
        solver_status::invalid_problem);
}

TEST(FindRoot, MatchesTheLegacyBoolForms)
{
    const auto f      = [](double x) { return std::cos(x) - x; };
    double     legacy = 0.0;
    ASSERT_TRUE(root_finding_algorithms::brent(f, 0.0, 1.0, legacy));
    const auto result = find_root(root_method::brent, scalar_function(f), 0.0, 1.0);
    ASSERT_TRUE(result.converged());
    EXPECT_DOUBLE_EQ(result.root, legacy);

    // The legacy form still leaves the output untouched when it does not converge.
    double               untouched     = 123.0;
    root_finding_options one_iteration = root_finding_options_builder()
                                             .with_max_iterations(1)
                                             .with_tolerance_function(0.0)
                                             .with_tolerance_parameter(0.0)
                                             .build();
    EXPECT_FALSE(root_finding_algorithms::bisection(f, 0.0, 1.0, untouched, one_iteration));
    EXPECT_DOUBLE_EQ(untouched, 123.0);
}

// -- real polynomial roots -----------------------------------------------------------------
void expect_roots(const real_roots_result& result, std::vector<double> expected, double tol = 1e-9)
{
    ASSERT_TRUE(result.converged()) << result.message;
    ASSERT_EQ(result.roots.size(), expected.size());
    std::sort(expected.begin(), expected.end());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result.roots[i], expected[i], tol * (1.0 + std::abs(expected[i])));
    }
}

TEST(RealRoots, Quadratic)
{
    expect_roots(real_roots_quadratic(-3.0, 2.0), {1.0, 2.0});
    expect_roots(real_roots_quadratic(-2.0, 1.0), {1.0});  // double root, once
    expect_roots(real_roots_quadratic(0.0, 1.0), {});      // no real roots
    // Cancellation-prone: roots ~1e8 and ~1e-8.
    expect_roots(real_roots_quadratic(-1e8, 1.0), {1e8, 1e-8}, 1e-7);
}

TEST(RealRoots, Cubic)
{
    expect_roots(real_roots_cubic(-6.0, 11.0, -6.0), {1.0, 2.0, 3.0});
    expect_roots(real_roots_cubic(0.0, 1.0, 1.0), {-0.6823278038280193});  // one real root
    expect_roots(real_roots_cubic(-6.0, 12.0, -8.0), {2.0});               // (x-2)^3
    expect_roots(real_roots_cubic(0.0, -3.0, 2.0), {-2.0, 1.0});           // (x-1)^2 (x+2)
}

TEST(RealRoots, Quartic)
{
    expect_roots(real_roots_quartic(0.0, -5.0, 0.0, 4.0), {-2.0, -1.0, 1.0, 2.0});
}

TEST(RealRoots, QuarticCases)
{
    // (x^2 + 1)(x - 1)(x - 3) = x^4 - 4x^3 + 4x^2 - 4x + 3: two real roots.
    expect_roots(real_roots_quartic(-4.0, 4.0, -4.0, 3.0), {1.0, 3.0});
    // (x^2 + 1)(x^2 + 4): none.
    expect_roots(real_roots_quartic(0.0, 5.0, 0.0, 4.0), {});
    // (x - 1)^2 (x^2 + 1) = x^4 - 2x^3 + 2x^2 - 2x + 1: a double real root.
    expect_roots(real_roots_quartic(-2.0, 2.0, -2.0, 1.0), {1.0});
    // (x - 2)^4.
    expect_roots(real_roots_quartic(-8.0, 24.0, -32.0, 16.0), {2.0}, 1e-3);
}

TEST(RealRoots, RecoversRandomRealRootSets)
{
    std::mt19937                           rng(11);
    std::uniform_real_distribution<double> position(-5.0, 5.0);
    for (int degree = 2; degree <= 7; ++degree)
    {
        for (int trial = 0; trial < 20; ++trial)
        {
            std::vector<double> truth;
            while (static_cast<int>(truth.size()) < degree)
            {
                const double candidate = position(rng);
                if (std::all_of(truth.begin(),
                        truth.end(),
                        [candidate](double r) { return std::abs(r - candidate) > 0.25; }))
                {
                    truth.push_back(candidate);
                }
            }
            std::vector<double> coefficients = {1.0};
            for (const double r : truth)
            {
                std::vector<double> next(coefficients.size() + 1, 0.0);
                for (std::size_t i = 0; i < coefficients.size(); ++i)
                {
                    next[i] += coefficients[i];
                    next[i + 1] -= r * coefficients[i];
                }
                coefficients = next;
            }
            SCOPED_TRACE("degree " + std::to_string(degree) + " trial " + std::to_string(trial));
            expect_roots(real_roots(coefficients), truth, 1e-7);
        }
    }
}

TEST(RealRoots, InvalidInputIsReported)
{
    EXPECT_EQ(real_roots({}).status, solver_status::invalid_problem);
    EXPECT_EQ(real_roots({5.0}).status, solver_status::invalid_problem);
    EXPECT_EQ(real_roots({0.0, 1.0, 2.0}).status, solver_status::invalid_problem);
    EXPECT_EQ(real_roots({1.0, kNaN, 2.0}).status, solver_status::invalid_problem);
}

TEST(RealRoots, SelectionPoliciesAreSeparateAndExplicit)
{
    const std::vector<double> roots = {-3.0, -1.0, 0.5, 2.0, 7.0};
    EXPECT_DOUBLE_EQ(*smallest_positive_root(roots), 0.5);
    EXPECT_DOUBLE_EQ(*largest_root(roots), 7.0);
    EXPECT_EQ(roots_in_interval(roots, -1.0, 2.0), (std::vector<double>{-1.0, 0.5, 2.0}));
    EXPECT_DOUBLE_EQ(*smallest_root_in_interval(roots, 0.0, 10.0), 0.5);
    EXPECT_FALSE(smallest_positive_root({-2.0, -1.0}).has_value());
    EXPECT_FALSE(smallest_root_in_interval(roots, 3.0, 6.0).has_value());
    EXPECT_FALSE(largest_root({}).has_value());

    // Clamping is a separate, opt-in step: nothing is raised unless asked.
    EXPECT_DOUBLE_EQ(*at_least(std::optional<double>(0.5), 2.0), 2.0);
    EXPECT_DOUBLE_EQ(*at_least(std::optional<double>(5.0), 2.0), 5.0);
    EXPECT_FALSE(at_least(std::nullopt, 2.0).has_value());
}

TEST(RealRoots, AgreesWithTheLegacyQuarticOnItsDocumentedCase)
{
    // (x - 1)(x - 2)(x + 3)(x + 4): the legacy solver returns the largest real
    // root clamped to the threshold, which is exactly largest_root + at_least.
    const double a3 = 4.0, a2 = -7.0, a1 = -22.0, a0 = 24.0;
    const auto   roots = real_roots_quartic(a3, a2, a1, a0);
    ASSERT_TRUE(roots.converged());
    const double legacy = polynomial_solver::fourth_degree_polynomial_solver(a3, a2, a1, a0, 0.0);
    EXPECT_NEAR(*at_least(largest_root(roots.roots), 0.0), legacy, 1e-9);
    expect_roots(roots, {-4.0, -3.0, 1.0, 2.0});
}
}  // namespace
}  // namespace solverslib::api
