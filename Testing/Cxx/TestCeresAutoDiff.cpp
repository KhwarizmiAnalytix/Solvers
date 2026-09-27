#include <cmath>
#include <gtest/gtest.h>

#include "detail/support.h"
#include "optimization_test_problems.h"

#if defined(SOLVERS_HAS_CERES)

#include "solvers/integrations/autodiff_provider.h"

namespace solverslib
{
namespace
{
using testing::optimization_test_problem;

// Test that auto_diff() creates a valid problem with provider
TEST(CeresAutoDiffIntegration, ProblemCreation)
{
    const auto& tp = testing::make_rosenbrock_problem();
    auto problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    EXPECT_EQ(problem.num_parameters, tp.num_parameters);
    EXPECT_EQ(problem.num_residuals, tp.num_residuals);
    EXPECT_TRUE(problem.provider_factory);
    EXPECT_TRUE(problem.has_jacobian_provider());
}

// Test that the provider metadata is correct
TEST(CeresAutoDiffIntegration, ProviderMetadata)
{
    const auto& tp = testing::make_rosenbrock_problem();
    auto problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    ASSERT_TRUE(problem.provider_factory);
    const auto& meta = problem.provider_factory->metadata();

    EXPECT_EQ(meta.num_parameters, tp.num_parameters);
    EXPECT_EQ(meta.num_residuals, tp.num_residuals);
    EXPECT_EQ(meta.source, api::derivative_mode::automatic_differentiation);
    EXPECT_TRUE(meta.supports_ceres);
}

// Test that evaluator can be created
TEST(CeresAutoDiffIntegration, EvaluatorCreation)
{
    const auto& tp = testing::make_rosenbrock_problem();
    auto problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    ASSERT_TRUE(problem.provider_factory);
    auto evaluator = problem.provider_factory->create_evaluator();

    EXPECT_TRUE(evaluator);
    EXPECT_EQ(evaluator->metadata().num_parameters, tp.num_parameters);
    EXPECT_EQ(evaluator->metadata().num_residuals, tp.num_residuals);
}

// Test that evaluator computes residuals correctly (double instantiation)
TEST(CeresAutoDiffIntegration, ResidualEvaluation)
{
    const double x_vals[] = {1.5, 0.5};
    const auto& tp = testing::make_rosenbrock_problem();

    auto problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    auto evaluator = problem.provider_factory->create_evaluator();

    vector_type x = to_vector_type(std::vector<double>(x_vals, x_vals + 2));
    vector_type r = make_vector(2);
    vector_type r_expected = make_vector(2);

    // Get expected residuals from the test problem
    tp.residuals(x, r_expected);

    // Evaluate through AD provider
    auto status = evaluator->evaluate(x, r);

    EXPECT_EQ(status, api::detail::evaluation_status::ok);
    EXPECT_NEAR(r[0], r_expected[0], 1e-14);
    EXPECT_NEAR(r[1], r_expected[1], 1e-14);
}

// Test that evaluator computes Jacobians via AD
TEST(CeresAutoDiffIntegration, JacobianEvaluation)
{
    const double x_vals[] = {1.5, 0.5};
    const auto& tp = testing::make_rosenbrock_problem();

    auto problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    auto evaluator = problem.provider_factory->create_evaluator();

    vector_type x = to_vector_type(std::vector<double>(x_vals, x_vals + 2));
    vector_type r = make_vector(2);
    matrix_type J = make_matrix(2, 2);
    matrix_type J_expected = make_matrix(2, 2);

    // Get expected Jacobian from the test problem
    tp.residuals(x, r);
    tp.jacobian(x, J_expected);

    // Evaluate residuals and Jacobian through AD provider
    auto status = evaluator->evaluate(x, r, &J);

    EXPECT_EQ(status, api::detail::evaluation_status::ok);

    // Check Jacobian entries against analytic derivative
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(J(i, j), J_expected(i, j), 1e-10)
                << "Jacobian mismatch at (" << i << ", " << j << ")";
        }
    }
}

// Test with more parameters (stride-4 test: 4 parameters, 4 residuals)
TEST(CeresAutoDiffIntegration, StrideMultiplePassesFourParams)
{
    const auto& tp = testing::make_powell_singular_problem();
    auto problem = least_squares(testing::PowellSingularResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    auto evaluator = problem.provider_factory->create_evaluator();

    vector_type x = to_vector_type(std::vector<double>{3.0, -1.0, 0.0, 1.0});
    vector_type r = make_vector(tp.num_residuals);
    matrix_type J = make_matrix(tp.num_residuals, tp.num_parameters);
    matrix_type J_expected = make_matrix(tp.num_residuals, tp.num_parameters);

    // Get expected Jacobian
    tp.residuals(x, r);
    tp.jacobian(x, J_expected);

    // Evaluate through AD provider
    auto status = evaluator->evaluate(x, r, &J);

    EXPECT_EQ(status, api::detail::evaluation_status::ok);

    // Verify Jacobian accuracy
    for (std::size_t i = 0; i < tp.num_residuals; ++i)
    {
        for (std::size_t j = 0; j < tp.num_parameters; ++j)
        {
            EXPECT_NEAR(J(i, j), J_expected(i, j), 1e-10)
                << "Jacobian mismatch at (" << i << ", " << j << ")";
        }
    }
}

// Test residual-only evaluation (jacobians=nullptr) should work
TEST(CeresAutoDiffIntegration, ResidualOnlyEvaluation)
{
    const double x_vals[] = {1.5, 0.5};
    const auto& tp = testing::make_rosenbrock_problem();

    auto problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    problem.derivatives(auto_diff());

    auto evaluator = problem.provider_factory->create_evaluator();

    vector_type x = to_vector_type(std::vector<double>(x_vals, x_vals + 2));
    vector_type r = make_vector(2);
    vector_type r_expected = make_vector(2);

    tp.residuals(x, r_expected);

    // Evaluate residuals only (no Jacobian)
    auto status = evaluator->evaluate(x, r, nullptr);

    EXPECT_EQ(status, api::detail::evaluation_status::ok);
    EXPECT_NEAR(r[0], r_expected[0], 1e-14);
    EXPECT_NEAR(r[1], r_expected[1], 1e-14);
}

// Tests above cover: Rosenbrock (2 params, 2 residuals), Powell (4 params, 4 residuals),
// residual-only evaluation, and Jacobian evaluation with multiple Jet passes.

// Note: Testing functor that returns false is complex with Ceres AD because
// the functor must be compatible with both double and Jet types in a const context.
// The core AD functionality is tested above.

}  // namespace
}  // namespace solverslib

#endif  // SOLVERS_HAS_CERES
