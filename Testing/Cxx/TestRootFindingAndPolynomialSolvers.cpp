#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "solvers/polynomial_solver.h"
#include "solvers/root_finding_algorithms.h"

namespace solverslib
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// root_finding_algorithms::brent
// ---------------------------------------------------------------------------

TEST(RootFindingBrent, FindsRootOfQuadratic)
{
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent(
        [](double x) { return x * x - 4.0; }, 0.0, 3.0, root, 0.0, 1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 2.0, 1e-9);
}

TEST(RootFindingBrent, FindsRootOfTranscendentalFixedPoint)
{
    // cos(x) - x == 0 is the classic fixed-point benchmark, root ~ 0.7390851332.
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent(
        [](double x) { return std::cos(x) - x; }, 0.0, 1.0, root, 0.0, 1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 0.7390851332, 1e-8);
}

TEST(RootFindingBrent, FindsRootOfCubic)
{
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent(
        [](double x) { return x * x * x - x - 2.0; }, 1.0, 2.0, root, 0.0, 1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 1.5213797068, 1e-8);
}

TEST(RootFindingBrent, HonorsFunctionOffset)
{
    // Solves cos(x) == 0.5 by passing f_0 rather than folding it into the lambda.
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent(
        [](double x) { return std::cos(x); }, 0.0, 1.2, root, 0.5, 1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, kPi / 3.0, 1e-8);
}

TEST(RootFindingBrent, ThrowsWhenBracketDoesNotContainRoot)
{
    double root = 0.0;
    EXPECT_THROW(
        root_finding_algorithms::brent([](double x) { return x * x - 4.0; }, 3.0, 5.0, root),
        std::invalid_argument);
}

TEST(RootFindingBrent, ReturnsFalseWhenIterationBudgetIsExhausted)
{
    // Machine-epsilon tolerances with only a handful of iterations cannot converge.
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent([](double x) { return x * x - 4.0; },
        0.0,
        3.0,
        root,
        0.0,
        std::numeric_limits<double>::epsilon(),
        std::numeric_limits<double>::epsilon(),
        2);

    EXPECT_FALSE(converged);
}

// ---------------------------------------------------------------------------
// root_finding_algorithms::dekker
// ---------------------------------------------------------------------------

TEST(RootFindingDekker, FindsRootOfQuadratic)
{
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::dekker(
        [](double x, double& df_dx)
        {
            df_dx = 2.0 * x;
            return x * x - 4.0;
        },
        0.0,
        3.0,
        root,
        1e-12,
        1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 2.0, 1e-9);
}

TEST(RootFindingDekker, FindsRootOfCubic)
{
    // x^3 - 2x - 5 == 0, root ~ 2.0945514815.
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::dekker(
        [](double x, double& df_dx)
        {
            df_dx = 3.0 * x * x - 2.0;
            return x * x * x - 2.0 * x - 5.0;
        },
        2.0,
        3.0,
        root,
        1e-12,
        1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 2.0945514815, 1e-8);
}

TEST(RootFindingDekker, ConvergesEarlyWhenBracketEndpointIsExactRoot)
{
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::dekker(
        [](double x, double& df_dx)
        {
            df_dx = 2.0 * x;
            return x * x - 4.0;
        },
        2.0,
        3.0,
        root);

    EXPECT_TRUE(converged);
    EXPECT_DOUBLE_EQ(root, 2.0);
}

TEST(RootFindingDekker, ThrowsWhenBracketDoesNotContainRoot)
{
    double root = 0.0;
    EXPECT_THROW(root_finding_algorithms::dekker(
                     [](double x, double& df_dx)
                     {
                         df_dx = 2.0 * x;
                         return x * x - 4.0;
                     },
                     3.0,
                     5.0,
                     root),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// polynomial_solver::second_degree_polynomial_solver
// ---------------------------------------------------------------------------

TEST(PolynomialSolverSecondDegree, ReturnsSmallestPositiveRootWhenBothRootsPositive)
{
    // x^2 - 5x + 6 == 0 -> roots {2, 3}.
    EXPECT_NEAR(polynomial_solver::second_degree_polynomial_solver(-5.0, 6.0), 2.0, 1e-12);
}

TEST(PolynomialSolverSecondDegree, ReturnsNanForExactlyRepeatedRoot)
{
    // x^2 - 2x + 1 == 0 -> double root at 1, i.e. discriminant == 0 exactly.
    // The implementation only branches on delta > 0, so an exact repeated
    // root falls through to the same NaN path as "no real roots".
    EXPECT_TRUE(std::isnan(polynomial_solver::second_degree_polynomial_solver(-2.0, 1.0)));
}

TEST(PolynomialSolverSecondDegree, FallsBackToLargerRootWhenNeitherIsPositive)
{
    // x^2 + 3x + 2 == 0 -> roots {-1, -2}; no positive root exists so the
    // implementation falls back to the larger (less negative) root.
    EXPECT_NEAR(polynomial_solver::second_degree_polynomial_solver(3.0, 2.0), -1.0, 1e-12);
}

TEST(PolynomialSolverSecondDegree, ReturnsNanWhenDiscriminantIsNegative)
{
    // x^2 + 1 == 0 has no real roots.
    EXPECT_TRUE(std::isnan(polynomial_solver::second_degree_polynomial_solver(0.0, 1.0)));
}

// ---------------------------------------------------------------------------
// polynomial_solver::third_degree_polynomial_solver
// ---------------------------------------------------------------------------

double cubic_residual(double b, double c, double d, double root)
{
    return d + root * (c + root * (b + root));
}

TEST(PolynomialSolverThirdDegree, SolvesThreeRealRootCase)
{
    // x^3 - 6x^2 + 11x - 6 == 0 -> roots {1, 2, 3}; verify the returned root
    // actually satisfies the equation rather than pinning a specific branch.
    const double root = polynomial_solver::third_degree_polynomial_solver(-6.0, 11.0, -6.0);
    EXPECT_NEAR(cubic_residual(-6.0, 11.0, -6.0, root), 0.0, 1e-9);
}

TEST(PolynomialSolverThirdDegree, SolvesSingleRealRootCase)
{
    // x^3 + 3x + 5 == 0 has exactly one real root, ~ -1.1541717.
    const double root = polynomial_solver::third_degree_polynomial_solver(0.0, 3.0, 5.0);
    EXPECT_NEAR(root, -1.1541717, 1e-6);
    EXPECT_NEAR(cubic_residual(0.0, 3.0, 5.0, root), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// polynomial_solver::fourth_degree_polynomial_solver
// ---------------------------------------------------------------------------

double quartic_residual(double a3, double a2, double a1, double a0, double root)
{
    return a0 + root * (a1 + root * (a2 + root * (a3 + root)));
}

TEST(PolynomialSolverFourthDegree, SolvesBiquadraticWithFourRealRoots)
{
    // x^4 - 10x^2 + 9 == 0 -> (x^2 - 1)(x^2 - 9) -> roots {-3, -1, 1, 3}.
    // The solver returns the largest real root.
    const double root = polynomial_solver::fourth_degree_polynomial_solver(0.0, -10.0, 0.0, 9.0);
    EXPECT_NEAR(root, 3.0, 1e-9);
    EXPECT_NEAR(quartic_residual(0.0, -10.0, 0.0, 9.0, root), 0.0, 1e-6);
}

TEST(PolynomialSolverFourthDegree, ClampsToThresholdWhenRootIsBelowIt)
{
    const double root =
        polynomial_solver::fourth_degree_polynomial_solver(0.0, -10.0, 0.0, 9.0, 10.0);
    EXPECT_DOUBLE_EQ(root, 10.0);
}

// ---------------------------------------------------------------------------
// Cross-checks between the closed-form solvers and the iterative root finders,
// confirming both families agree on the same underlying polynomial.
// ---------------------------------------------------------------------------

TEST(PolynomialAndRootFindingConsistency, AgreeOnQuadraticRoot)
{
    const double closed_form = polynomial_solver::second_degree_polynomial_solver(-5.0, 6.0);

    double iterative_root = 0.0;
    ASSERT_TRUE(root_finding_algorithms::brent(
        [](double x) { return x * x - 5.0 * x + 6.0; }, 1.5, 2.5, iterative_root, 0.0, 1e-12));

    EXPECT_NEAR(closed_form, iterative_root, 1e-8);
}

}  // namespace
}  // namespace solverslib
