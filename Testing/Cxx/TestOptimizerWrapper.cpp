/**
 * @file TestOptimizerWrapper.cpp
 * @brief Simple test for the solver_wrapper class
 *
 * Tests demonstrate how the wrapper eliminates switch statements and provides
 * a unified interface for all optimization algorithms.
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "common/pointer.h"
#include "solvers/solver_options/solver_options_lm.h"
#include "solvers/solver_wrapper.h"
#include "quarismaTest.h"
#include "quarismaTesting.h"

namespace solverslib
{

class TestOptimizerWrapper
{
public:
    void SetUp()
    {
        // Setup test problem: minimize Rosenbrock function
        // f(x,y) = (a-x)^2 + b(y-x^2)^2, with a=1, b=100
        // Minimum at (1,1) with f(1,1) = 0

        a_              = 1.0;
        b_              = 100.0;
        num_parameters_ = 2;
        num_residuals_  = 2;

        // Initial guess
        initial_params_ = {-1.2, 1.0};

        // Expected solution
        expected_solution_ = {1.0, 1.0};
        tolerance_         = 1e-6;

        // Bounds
        lower_bounds_ = {-2.0, -2.0};
        upper_bounds_ = {2.0, 2.0};
    }

    // Rosenbrock function as residual-based objective
    solver_wrapper::objective_function_type create_rosenbrock_residual_objective()
    {
        return [this](const vector<double>& x, vector<double>& residuals)
        {
            // residuals should already be sized correctly by caller
            if (residuals.size() >= 2)
            {
                residuals[0] = std::sqrt(a_ - x[0]);
                residuals[1] = std::sqrt(b_) * (x[1] - x[0] * x[0]);
            }
        };
    }

    // AAD jacobian for residual-based objective
    solver_wrapper::objective_function_aad_type create_rosenbrock_jacobian()
    {
        return [this](const vector<double>& x, matrix<double>& jacobian)
        {
            // jacobian should already be sized correctly by caller
            if (jacobian.rows() >= 2 && jacobian.columns() >= 2)
            {
                jacobian[0][0] = -0.5 / std::sqrt(a_ - x[0]);
                jacobian[0][1] = 0.0;
                jacobian[1][0] = -2.0 * std::sqrt(b_) * x[0];
                jacobian[1][1] = std::sqrt(b_);
            }
        };
    }

    void verify_solution(const std::vector<double>& solution)
    {
        //(solution[0], expected_solution_[0], tolerance_);
        EXPECT_NEAR(solution[1], expected_solution_[1], tolerance_);
    }

    void test_solver_wrapper_basic()
    {
        // Simple quadratic function: f(x) = (x-1)^2, minimum at x=1
        size_t num_parameters = 1;
        size_t num_residuals  = 1;

        auto objective = [](const vector<double>& x, vector<double>& residuals)
        {
            residuals[0] = x[0] - 1.0;  // Residual for (x-1)^2
        };

        auto optimizer = util::make_ptr_unique_const<solverslib::solver_wrapper>(
            num_parameters, num_residuals, objective);

        // Test with LM solver (always available)
        auto options = util::make_ptr_const<solver_options_lm>(100, 1e-6);

        std::vector<double> params  = {0.0};  // Initial guess
        bool                success = optimizer->solve(params, options);

        QUARISMA_LOGF(
            INFO,
            "Optimizer wrapper test: success=%s, result=%f",
            success ? "true" : "false",
            params[0]);

        // Should converge to x=1
        EXPECT_TRUE(success);
        EXPECT_NEAR(params[0], 1.0, 1e-3);
    }

    void test_optimizer_support()
    {
        // Test optimizer support detection
        EXPECT_TRUE(solver_wrapper::is_supported(solver_enum::LM));

        bool ceres_supported = solver_wrapper::is_supported(solver_enum::CERES);
        bool nlopt_supported = solver_wrapper::is_supported(solver_enum::NLOPT);

        QUARISMA_LOGF(
            INFO,
            "Optimizer support: LM=true, CERES=%s, NLOPT=%s",
            ceres_supported ? "true" : "false",
            nlopt_supported ? "true" : "false");
    }

private:
    double              a_, b_;
    size_t              num_parameters_, num_residuals_;
    std::vector<double> initial_params_;
    std::vector<double> expected_solution_;
    std::vector<double> lower_bounds_, upper_bounds_;
    double              tolerance_;
};
}  // namespace solverslib

QUARISMATEST(Math, OptimizerWrapper)
{
    solverslib::TestOptimizerWrapper test;
    test.SetUp();

    QUARISMA_LOGF(INFO, "=== Testing Optimizer Wrapper ===");

    test.test_solver_wrapper_basic();
    QUARISMA_LOGF(INFO, "✓ Basic optimizer wrapper test passed");

    test.test_optimizer_support();
    QUARISMA_LOGF(INFO, "✓ Optimizer support test passed");

    QUARISMA_LOGF(INFO, "=== All Optimizer Wrapper Tests Passed! ===");

    END_TEST();
}
