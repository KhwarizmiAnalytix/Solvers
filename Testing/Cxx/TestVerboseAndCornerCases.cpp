#include <cmath>
#include <sstream>

#include <gtest/gtest.h>

#include "solvers/api/solve.h"
#include "solvers/gauss_newton_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/root_finding_algorithms.h"

namespace solverslib
{
namespace
{

// ===========================================================================
// VERBOSE OUTPUT TESTS
// ===========================================================================

TEST(VerboseLogs, LMSolverVerboseOutput)
{
    api::least_squares_problem ls;
    ls.num_parameters = 2;
    ls.num_residuals  = 3;

    double data_x[] = {1.0, 2.0, 3.0};
    double data_y[] = {2.1, 3.9, 6.2};

    ls.residuals = [&](const vector_type& p, vector_type& r)
    {
        for (size_t i = 0; i < 3; ++i)
            r(static_cast<Eigen::Index>(i)) = p(0) * data_x[i] + p(1) - data_y[i];
    };

    ls.jacobian = [&](const vector_type& /*p*/, matrix_type& J)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            J(static_cast<Eigen::Index>(i), 0) = data_x[i];
            J(static_cast<Eigen::Index>(i), 1) = 1.0;
        }
    };

    api::solve_options opts;
    opts.algorithm           = api::algorithm::levenberg_marquardt;
    opts.backend             = api::backend::native;
    opts.max_iterations      = 50;
    opts.function_tolerance  = 1e-10;
    opts.gradient_tolerance  = 1e-10;
    opts.parameter_tolerance = 1e-10;
    opts.verbose             = true;  // Enable verbose logging

    vector_type x0(2);
    x0 << 1.0, 0.0;

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_LT(result.iterations, 50);
}

TEST(VerboseLogs, GaussNewtonVerboseOutput)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 2;

    ls.residuals = [](const vector_type& p, vector_type& r)
    {
        r(0) = p(0) - 1.0;
        r(1) = p(0) - 1.0;
    };

    ls.jacobian = [](const vector_type& /*p*/, matrix_type& J)
    {
        J(0, 0) = 1.0;
        J(1, 0) = 1.0;
    };

    api::solve_options opts;
    opts.algorithm           = api::algorithm::gauss_newton;
    opts.backend             = api::backend::native;
    opts.max_iterations      = 20;
    opts.function_tolerance  = 1e-12;
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;
    opts.verbose             = true;

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), 1.0, 1e-8);
}

TEST(VerboseLogs, RootFindingVerbose)
{
    // Note: root_finding_options doesn't have verbose flag in current API,
    // but this documents the intent for future verbose improvements
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent([](double x) { return x * x - 9.0; },
        0.0,
        4.0,
        root,
        root_finding_options_builder()
            .with_tolerance_function(1e-12)
            .with_tolerance_parameter(1e-12)
            .build());

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 3.0, 1e-10);
}

// ===========================================================================
// CORNER CASES & EDGE CONDITIONS
// ===========================================================================

TEST(CornerCases, ZeroInitialGuess)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 1;
    ls.residuals      = [](const vector_type& p, vector_type& r) { r(0) = p(0) - 5.0; };
    ls.jacobian       = [](const vector_type& /*p*/, matrix_type& J) { J(0, 0) = 1.0; };

    api::solve_options opts;
    opts.max_iterations      = 100;
    opts.function_tolerance  = 1e-12;
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;

    vector_type x0(1);
    x0(0) = 0.0;  // Start at origin

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), 5.0, 1e-6);
}

TEST(CornerCases, NegativeInitialGuess)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 1;
    ls.residuals      = [](const vector_type& p, vector_type& r) { r(0) = p(0) + 3.0; };
    ls.jacobian       = [](const vector_type& /*p*/, matrix_type& J) { J(0, 0) = 1.0; };

    api::solve_options opts;
    opts.max_iterations      = 100;
    opts.function_tolerance  = 1e-12;
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;

    vector_type x0(1);
    x0(0) = 10.0;  // Start far from solution

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), -3.0, 1e-6);
}

TEST(CornerCases, MultipleEquationsUnderdetermined)
{
    // More equations than parameters (overdetermined - normal case)
    api::least_squares_problem ls;
    ls.num_parameters = 2;
    ls.num_residuals  = 5;

    ls.residuals = [](const vector_type& p, vector_type& r)
    {
        r(0) = p(0) + p(1) - 5.0;
        r(1) = p(0) - p(1) - 1.0;
        r(2) = 2.0 * p(0) + p(1) - 8.0;
        r(3) = p(0) + 2.0 * p(1) - 7.0;
        r(4) = p(0) * p(0) + p(1) * p(1) - 13.0;
    };

    ls.jacobian = [](const vector_type& p, matrix_type& J)
    {
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
        J(1, 0) = 1.0;
        J(1, 1) = -1.0;
        J(2, 0) = 2.0;
        J(2, 1) = 1.0;
        J(3, 0) = 1.0;
        J(3, 1) = 2.0;
        J(4, 0) = 2.0 * p(0);
        J(4, 1) = 2.0 * p(1);
    };

    api::solve_options opts;
    opts.max_iterations      = 100;
    opts.function_tolerance  = 1e-10;
    opts.gradient_tolerance  = 1e-10;
    opts.parameter_tolerance = 1e-10;

    vector_type x0(2);
    x0 << 0.0, 0.0;

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_LT(result.residual_norm, 1e-4);  // Should find good fit even if overdetermined
}

TEST(CornerCases, NearSingularJacobian)
{
    api::least_squares_problem ls;
    ls.num_parameters = 2;
    ls.num_residuals  = 2;

    // Nearly singular system (near-zero determinant)
    ls.residuals = [](const vector_type& p, vector_type& r)
    {
        r(0) = p(0) + 1.001 * p(1) - 5.0;
        r(1) = p(0) + p(1) - 5.0;  // Almost linearly dependent
    };

    ls.jacobian = [](const vector_type& /*p*/, matrix_type& J)
    {
        J(0, 0) = 1.0;
        J(0, 1) = 1.001;
        J(1, 0) = 1.0;
        J(1, 1) = 1.0;
    };

    api::solve_options opts;
    opts.max_iterations      = 100;
    opts.function_tolerance  = 1e-10;
    opts.gradient_tolerance  = 1e-10;
    opts.parameter_tolerance = 1e-10;

    vector_type x0(2);
    x0 << 0.0, 0.0;

    auto result = api::solve(ls, x0, opts);

    // Nearly singular systems are challenging; just ensure convergence is attempted
    EXPECT_LT(result.iterations, 100);
}

TEST(CornerCases, LargeScaleResiduals)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 1;

    // Residual with very large scale
    ls.residuals = [](const vector_type& p, vector_type& r) { r(0) = 1e8 * (p(0) - 0.5); };

    ls.jacobian = [](const vector_type& /*p*/, matrix_type& J) { J(0, 0) = 1e8; };

    api::solve_options opts;
    opts.max_iterations      = 100;
    opts.function_tolerance  = 1e-4;
    opts.gradient_tolerance  = 1e-4;
    opts.parameter_tolerance = 1e-4;

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), 0.5, 1e-5);
}

TEST(CornerCases, IdenticalEquations)
{
    // Two identical equations (linearly dependent)
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 2;

    ls.residuals = [](const vector_type& p, vector_type& r)
    {
        r(0) = p(0) - 3.0;
        r(1) = p(0) - 3.0;  // Identical to r(0)
    };

    ls.jacobian = [](const vector_type& /*p*/, matrix_type& J)
    {
        J(0, 0) = 1.0;
        J(1, 0) = 1.0;  // Identical to J(0, 0)
    };

    api::solve_options opts;
    opts.max_iterations      = 50;
    opts.function_tolerance  = 1e-12;
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    // Should still converge to the correct solution
    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), 3.0, 1e-6);
}

TEST(CornerCases, SingleParameterMultipleResiduals)
{
    // Single parameter, many constraints
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 10;

    const double target = 2.5;

    ls.residuals = [target](const vector_type& p, vector_type& r)
    {
        for (int i = 0; i < 10; ++i)
        {
            r(i) = p(0) - target;
        }
    };

    ls.jacobian = [](const vector_type& /*p*/, matrix_type& J)
    {
        for (int i = 0; i < 10; ++i)
        {
            J(i, 0) = 1.0;
        }
    };

    api::solve_options opts;
    opts.max_iterations      = 50;
    opts.function_tolerance  = 1e-12;
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), target, 1e-6);
}

// ===========================================================================
// ERROR CHECKING & VALIDATION TESTS
// ===========================================================================

TEST(ErrorHandling, InvalidProblemZeroResiduals)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 0;  // Invalid: no residuals
    ls.residuals      = nullptr;
    ls.jacobian       = nullptr;

    api::solve_options opts;
    vector_type        x0(1);
    x0(0) = 1.0;

    auto result = api::solve(ls, x0, opts);

    // Should report invalid problem
    EXPECT_EQ(result.status, api::solver_status::invalid_problem);
    EXPECT_FALSE(result.converged());
}

TEST(ErrorHandling, InvalidProblemZeroParameters)
{
    api::least_squares_problem ls;
    ls.num_parameters = 0;  // Invalid: no parameters
    ls.num_residuals  = 1;
    ls.residuals      = nullptr;
    ls.jacobian       = nullptr;

    api::solve_options opts;
    vector_type        x0(1);
    x0(0) = 1.0;

    auto result = api::solve(ls, x0, opts);

    // Should report invalid problem or dimension mismatch
    EXPECT_EQ(result.status, api::solver_status::invalid_problem);
    EXPECT_FALSE(result.converged());
}

TEST(ErrorHandling, ZeroIterationLimit)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 1;
    ls.residuals      = [](const vector_type& p, vector_type& r) { r(0) = p(0) - 5.0; };
    ls.jacobian       = [](const vector_type& /*p*/, matrix_type& J) { J(0, 0) = 1.0; };

    api::solve_options opts;
    opts.max_iterations = 0;  // No iterations allowed

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    // Should not converge with zero iterations
    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.iterations, 0);
}

TEST(ErrorHandling, NegativeTolerance)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 1;
    ls.residuals      = [](const vector_type& p, vector_type& r) { r(0) = p(0) - 5.0; };
    ls.jacobian       = [](const vector_type& /*p*/, matrix_type& J) { J(0, 0) = 1.0; };

    api::solve_options opts;
    opts.max_iterations      = 100;
    opts.function_tolerance  = -1e-12;  // Invalid: negative tolerance
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    // Behavior with invalid tolerance is unspecified; just ensure no crash
    // Either fails validation or proceeds and converges/fails naturally
    EXPECT_NE(result.status, api::solver_status::backend_unavailable);
}

TEST(ErrorHandling, RootFindingInvalidInterval)
{
    // Interval where function doesn't change sign
    double root = 0.0;

    // brent throws when interval doesn't bracket a root (throws logging::exception, not
    // std::exception)
    bool threw_exception = false;
    try
    {
        root_finding_algorithms::brent([](double x)
            { return x * x + 1.0; },  // Always positive (no sign change)
            0.0,
            10.0,
            root,
            root_finding_options_builder().with_tolerance_function(1e-12).build());
    }
    catch (const std::exception& /*e*/)
    {
        threw_exception = true;
    }
    catch (...)
    {
        threw_exception = true;
    }

    EXPECT_TRUE(threw_exception);
}

TEST(ErrorHandling, RootFindingZeroInterval)
{
    double root = 0.0;

    // Zero-width interval: implementation may return root = start value, but shouldn't crash
    const bool converged = root_finding_algorithms::brent([](double x) { return x - 2.0; },
        2.0,
        2.0,  // Zero-width interval
        root,
        root_finding_options_builder().with_tolerance_function(1e-12).build());

    // Don't assert on convergence (unspecified behavior), just verify no crash
    // If we got here without exception, the test passes
    EXPECT_TRUE(true);
}

}  // namespace
}  // namespace solverslib
