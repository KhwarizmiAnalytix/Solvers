#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "solver_options/solver_options_lm.h"
#include "solver_wrapper.h"

namespace solverslib
{
namespace
{

TEST(OptimizerWrapper, SolvesVectorBackedObjective)
{
    solver_wrapper optimizer(
        1,
        1,
        [](const vector_type& parameters, vector_type& residuals) {
            residuals[0] = parameters[0] - 2.0;
        });
    std::shared_ptr<const solver_options> options =
        std::make_shared<solver_options_lm>(100, 1e-12, 1e-12, 1e-12);
    std::vector<double> parameters{0.0};

    EXPECT_TRUE(optimizer.solve(parameters, options));
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_NEAR(parameters[0], 2.0, 1e-6);
}

TEST(OptimizerWrapper, ReportsBuiltInSolverSupport)
{
    EXPECT_TRUE(solver_wrapper::is_supported(solver_enum::LM));
    EXPECT_TRUE(solver_wrapper::is_supported(solver_enum::LBFGS));
    EXPECT_FALSE(solver_wrapper::is_supported(static_cast<solver_enum>(-1)));
}

}  // namespace
}  // namespace solverslib
