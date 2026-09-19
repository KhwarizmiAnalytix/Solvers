#include <gtest/gtest.h>

#include "solver_options/solver_options_lm.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/root_finding_algorithms.h"

namespace solverslib
{
namespace
{

TEST(OptimizationAlgorithm, LevenbergMarquardtSolvesScalarResidual)
{
    levenberg_marquardt_solver solver(
        1,
        1,
        [](const vector_type& parameters, vector_type& residuals)
        { residuals[0] = parameters[0] - 3.0; },
        [](const vector_type&, matrix_type& jacobian) { jacobian(0, 0) = 1.0; });
    solver_options_lm options(100, 1e-12, 1e-12, 1e-12);
    vector_type        parameters(1);
    parameters[0] = 0.0;

    const solver_output result = solver.solve(parameters, options);

    EXPECT_NEAR(parameters[0], 3.0, 1e-6);
    EXPECT_NE(result.status_, solver_convergence_enum::NOT_CONVERGED);
}

TEST(OptimizationAlgorithm, BrentFindsRoot)
{
    double     root      = 0.0;
    const bool converged = root_finding_algorithms::brent(
        [](double value) { return value * value - 4.0; },
        0.0,
        3.0,
        root,
        root_finding_options_builder().with_tolerance_function(1e-12).build());

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 2.0, 1e-10);
}

}  // namespace
}  // namespace solverslib
