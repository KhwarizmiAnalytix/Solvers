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

TEST(RootFinding, Solve)
{
    const auto options = tight_options();

    {
        SCOPED_TRACE("Bisection");
        const auto sqrt2 = run_bisection(f_x2m2, 1.0, 2.0, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);
        EXPECT_GT(sqrt2.iterations, 0u);

        const auto pi = run_bisection(f_sin, 3.0, 4.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        const scalar_function already = [](double x) { return x - 1.0; };
        const auto            hit     = run_bisection(already, 1.0, 2.0, options);
        EXPECT_EQ(hit.outcome, root_outcome::converged);
        EXPECT_DOUBLE_EQ(hit.root, 1.0);
        EXPECT_EQ(hit.iterations, 0u);

        const auto limited = run_bisection(
            f_x2m2, 1.0, 2.0, root_finding_options_builder().with_max_iterations(3).build());
        EXPECT_EQ(limited.outcome, root_outcome::iteration_limit);
        EXPECT_EQ(limited.iterations, 3u);

        EXPECT_THROW(run_bisection(f_x2m2, 2.0, 3.0, options), std::exception);
    }
    {
        SCOPED_TRACE("FalsePosition");
        const auto sqrt2 = run_false_position(f_x2m2, 1.0, 2.0, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);

        const auto pi = run_false_position(f_sin, 3.0, 4.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        EXPECT_THROW(run_false_position(f_x2m2, 2.0, 3.0, options), std::exception);
    }
    {
        SCOPED_TRACE("Ridders");
        const auto sqrt2 = run_ridders(f_x2m2, 1.0, 2.0, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);

        const auto pi = run_ridders(f_sin, 3.0, 4.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        EXPECT_THROW(run_ridders(f_x2m2, 2.0, 3.0, options), std::exception);
    }
    {
        SCOPED_TRACE("Brent");
        const auto sqrt2 = run_brent(f_x2m2, 1.0, 2.0, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);

        const auto pi = run_brent(f_sin, 3.0, 4.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        const scalar_function square = [](double x) { return x * x; };
        const auto            offset = run_brent(square,
            1.0,
            2.0,
            root_finding_options_builder()
                .with_tolerance_function(1e-12)
                .with_tolerance_parameter(1e-12)
                .with_max_iterations(100)
                .with_function_offset(3.0)
                .build());
        EXPECT_EQ(offset.outcome, root_outcome::converged);
        EXPECT_NEAR(offset.root, std::sqrt(3.0), kTol);

        const auto limited = run_brent(
            f_x2m2, 1.0, 2.0, root_finding_options_builder().with_max_iterations(2).build());
        EXPECT_EQ(limited.outcome, root_outcome::iteration_limit);

        EXPECT_THROW(run_brent(f_x2m2, 2.0, 3.0, options), std::exception);
    }
    {
        SCOPED_TRACE("Dekker");
        const auto sqrt2 = run_dekker(f_x2m2_grad, 1.0, 2.0, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);

        const auto pi = run_dekker(f_sin_grad, 3.0, 4.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        EXPECT_THROW(run_dekker(f_x2m2_grad, 2.0, 3.0, options), std::exception);
    }
    {
        SCOPED_TRACE("NewtonRaphson");
        const auto sqrt2 = run_newton_raphson(f_x2m2_grad, 1.5, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);

        const auto pi = run_newton_raphson(f_sin_grad, 3.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        const scalar_function_gradient flat = [](double x, double& df) -> double
        {
            df = 2.0 * x;
            return x * x - 4.0;
        };
        EXPECT_THROW(run_newton_raphson(flat, 0.0, options), std::exception);
    }
    {
        SCOPED_TRACE("Secant");
        const auto sqrt2 = run_secant(f_x2m2, 1.0, 2.0, options);
        EXPECT_EQ(sqrt2.outcome, root_outcome::converged);
        EXPECT_NEAR(sqrt2.root, std::sqrt(2.0), kTol);

        const auto pi = run_secant(f_sin, 3.0, 4.0, options);
        EXPECT_EQ(pi.outcome, root_outcome::converged);
        EXPECT_NEAR(pi.root, kPi, kTol);

        const auto limited = run_secant(
            f_x2m2, 1.0, 2.0, root_finding_options_builder().with_max_iterations(1).build());
        EXPECT_EQ(limited.outcome, root_outcome::iteration_limit);
    }
    {
        SCOPED_TRACE("Polynomial");
        EXPECT_NEAR(polynomial_solver::second_degree_polynomial_solver(-3.0, 2.0), 1.0, kTol);
        EXPECT_NEAR(
            polynomial_solver::second_degree_polynomial_solver(0.0, -2.0), std::sqrt(2.0), kTol);
        EXPECT_TRUE(std::isnan(polynomial_solver::second_degree_polynomial_solver(0.0, 1.0)));
        EXPECT_NEAR(polynomial_solver::third_degree_polynomial_solver(0.0, 0.0, -1.0), 1.0, kTol);
        EXPECT_NEAR(polynomial_solver::third_degree_polynomial_solver(-6.0, 11.0, -6.0), 3.0, 1e-9);
        EXPECT_NEAR(polynomial_solver::fourth_degree_polynomial_solver(-10.0, 35.0, -50.0, 24.0),
            4.0,
            1e-6);
        EXPECT_DOUBLE_EQ(
            polynomial_solver::fourth_degree_polynomial_solver(-10.0, 35.0, -50.0, 24.0, 5.0), 5.0);
    }
}
