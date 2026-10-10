#include <gtest/gtest.h>

#include <cmath>

#include "solver_options/root_finding_options.h"
#include "solvers/polynomial_solver.h"
#include "solvers/root_finding_algorithms.h"

using namespace solverslib;
using namespace solverslib::detail;

namespace
{
constexpr double kPi  = 3.141592653589793238462643383279502884;
constexpr double kTol = 1e-10;

root_finding_options tight_options()
{
    return root_finding_options_builder()
        .with_tolerance_function(1e-12)
        .with_tolerance_parameter(1e-12)
        .with_max_iterations(100)
        .build();
}

// f(x) = x² - 2,  root at √2
const scalar_function          f_x2m2      = [](double x) { return x * x - 2.0; };
const scalar_function_gradient f_x2m2_grad = [](double x, double& df) -> double
{
    df = 2.0 * x;
    return x * x - 2.0;
};

// f(x) = sin(x),  root at π ∈ [3, 4]
const scalar_function          f_sin      = [](double x) { return std::sin(x); };
const scalar_function_gradient f_sin_grad = [](double x, double& df) -> double
{
    df = std::cos(x);
    return std::sin(x);
};
}  // namespace

// ---------------------------------------------------------------------------
// Bisection
// ---------------------------------------------------------------------------

TEST(Bisection, ConvergesOnSqrt2)
{
    const auto r = run_bisection(f_x2m2, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
    EXPECT_GT(r.iterations, 0u);
}

TEST(Bisection, ConvergesOnPi)
{
    const auto r = run_bisection(f_sin, 3.0, 4.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(Bisection, HitsBracketEndpointImmediately)
{
    // x1 is already the root — should converge in 0 iterations
    const scalar_function f = [](double x) { return x - 1.0; };
    const auto            r = run_bisection(f, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_DOUBLE_EQ(r.root, 1.0);
    EXPECT_EQ(r.iterations, 0u);
}

TEST(Bisection, ReachesIterationLimit)
{
    auto       opts = root_finding_options_builder().with_max_iterations(3).build();
    const auto r    = run_bisection(f_x2m2, 1.0, 2.0, opts);
    EXPECT_EQ(r.outcome, root_outcome::iteration_limit);
    EXPECT_EQ(r.iterations, 3u);
}

TEST(Bisection, BadBracketThrows)
{
    EXPECT_THROW(run_bisection(f_x2m2, 2.0, 3.0, tight_options()), std::exception);
}

// ---------------------------------------------------------------------------
// False position (Illinois variant)
// ---------------------------------------------------------------------------

TEST(FalsePosition, ConvergesOnSqrt2)
{
    const auto r = run_false_position(f_x2m2, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
}

TEST(FalsePosition, ConvergesOnPi)
{
    const auto r = run_false_position(f_sin, 3.0, 4.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(FalsePosition, BadBracketThrows)
{
    EXPECT_THROW(run_false_position(f_x2m2, 2.0, 3.0, tight_options()), std::exception);
}

// ---------------------------------------------------------------------------
// Ridders
// ---------------------------------------------------------------------------

TEST(Ridders, ConvergesOnSqrt2)
{
    const auto r = run_ridders(f_x2m2, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
}

TEST(Ridders, ConvergesOnPi)
{
    const auto r = run_ridders(f_sin, 3.0, 4.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(Ridders, BadBracketThrows)
{
    EXPECT_THROW(run_ridders(f_x2m2, 2.0, 3.0, tight_options()), std::exception);
}

// ---------------------------------------------------------------------------
// Brent
// ---------------------------------------------------------------------------

TEST(Brent, ConvergesOnSqrt2)
{
    const auto r = run_brent(f_x2m2, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
}

TEST(Brent, ConvergesOnPi)
{
    const auto r = run_brent(f_sin, 3.0, 4.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(Brent, FunctionOffset)
{
    // solve x² = 3 via offset — same as f(x) = x² with target = 3
    const scalar_function f    = [](double x) { return x * x; };
    auto                  opts = root_finding_options_builder()
                    .with_tolerance_function(1e-12)
                    .with_tolerance_parameter(1e-12)
                    .with_max_iterations(100)
                    .with_function_offset(3.0)
                    .build();
    const auto r = run_brent(f, 1.0, 2.0, opts);
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(3.0), kTol);
}

TEST(Brent, ReachesIterationLimit)
{
    auto       opts = root_finding_options_builder().with_max_iterations(2).build();
    const auto r    = run_brent(f_x2m2, 1.0, 2.0, opts);
    EXPECT_EQ(r.outcome, root_outcome::iteration_limit);
}

TEST(Brent, BadBracketThrows)
{
    EXPECT_THROW(run_brent(f_x2m2, 2.0, 3.0, tight_options()), std::exception);
}

// ---------------------------------------------------------------------------
// Dekker
// ---------------------------------------------------------------------------

TEST(Dekker, ConvergesOnSqrt2)
{
    const auto r = run_dekker(f_x2m2_grad, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
}

TEST(Dekker, ConvergesOnPi)
{
    const auto r = run_dekker(f_sin_grad, 3.0, 4.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(Dekker, BadBracketThrows)
{
    EXPECT_THROW(run_dekker(f_x2m2_grad, 2.0, 3.0, tight_options()), std::exception);
}

// ---------------------------------------------------------------------------
// Newton-Raphson
// ---------------------------------------------------------------------------

TEST(NewtonRaphson, ConvergesOnSqrt2)
{
    const auto r = run_newton_raphson(f_x2m2_grad, 1.5, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
}

TEST(NewtonRaphson, ConvergesOnPi)
{
    const auto r = run_newton_raphson(f_sin_grad, 3.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(NewtonRaphson, VanishingDerivativeThrows)
{
    // f(x) = x² - 4, roots at ±2; derivative vanishes at x=0 while f(0)=-4 ≠ 0
    const scalar_function_gradient f = [](double x, double& df) -> double
    {
        df = 2.0 * x;
        return x * x - 4.0;
    };
    EXPECT_THROW(run_newton_raphson(f, 0.0, tight_options()), std::exception);
}

// ---------------------------------------------------------------------------
// Secant
// ---------------------------------------------------------------------------

TEST(Secant, ConvergesOnSqrt2)
{
    const auto r = run_secant(f_x2m2, 1.0, 2.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), kTol);
}

TEST(Secant, ConvergesOnPi)
{
    const auto r = run_secant(f_sin, 3.0, 4.0, tight_options());
    EXPECT_EQ(r.outcome, root_outcome::converged);
    EXPECT_NEAR(r.root, kPi, kTol);
}

TEST(Secant, ReachesIterationLimit)
{
    auto       opts = root_finding_options_builder().with_max_iterations(1).build();
    const auto r    = run_secant(f_x2m2, 1.0, 2.0, opts);
    EXPECT_EQ(r.outcome, root_outcome::iteration_limit);
}

// ---------------------------------------------------------------------------
// Polynomial solvers
// ---------------------------------------------------------------------------

TEST(SecondDegreePolynomial, MinimumPositiveRoot)
{
    // x² - 3x + 2 = 0 → roots 1, 2; min positive = 1
    EXPECT_NEAR(polynomial_solver::second_degree_polynomial_solver(-3.0, 2.0), 1.0, kTol);
}

TEST(SecondDegreePolynomial, Sqrt2Root)
{
    // x² - 2 = 0 → root at √2
    EXPECT_NEAR(
        polynomial_solver::second_degree_polynomial_solver(0.0, -2.0), std::sqrt(2.0), kTol);
}

TEST(SecondDegreePolynomial, NoRealRootsIsNaN)
{
    // x² + 1 = 0 → no real roots
    EXPECT_TRUE(std::isnan(polynomial_solver::second_degree_polynomial_solver(0.0, 1.0)));
}

TEST(ThirdDegreePolynomial, SingleRealRoot)
{
    // x³ - 1 = 0 → only real root is 1
    EXPECT_NEAR(polynomial_solver::third_degree_polynomial_solver(0.0, 0.0, -1.0), 1.0, kTol);
}

TEST(ThirdDegreePolynomial, ThreeRealRootsReturnsLargest)
{
    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6 → largest root = 3
    EXPECT_NEAR(polynomial_solver::third_degree_polynomial_solver(-6.0, 11.0, -6.0), 3.0, 1e-9);
}

TEST(FourthDegreePolynomial, LargestRoot)
{
    // (x-1)(x-2)(x-3)(x-4) = x⁴ - 10x³ + 35x² - 50x + 24 → largest real root = 4
    EXPECT_NEAR(
        polynomial_solver::fourth_degree_polynomial_solver(-10.0, 35.0, -50.0, 24.0), 4.0, 1e-6);
}

TEST(FourthDegreePolynomial, ThresholdClampsResult)
{
    // Same polynomial; threshold = 5 → returned value is 5
    EXPECT_DOUBLE_EQ(
        polynomial_solver::fourth_degree_polynomial_solver(-10.0, 35.0, -50.0, 24.0, 5.0), 5.0);
}
