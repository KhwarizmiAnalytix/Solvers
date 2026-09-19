#include <Eigen/Core>
#include <gtest/gtest.h>

#include "root_finding_algorithms.h"
#include "solver_options/solver_options_lm.h"
#include "solvers/levenberg_marquardt_solver.h"

namespace solverslib
{
namespace
{

TEST(OptimizationAlgorithm, LevenbergMarquardtSolvesScalarResidual)
{
    levenberg_marquardt_solver solver(
        1,
        1,
        [](const Eigen::VectorXd& parameters, Eigen::VectorXd& residuals) {
            residuals[0] = parameters[0] - 3.0;
        },
        [](const Eigen::VectorXd&, Eigen::MatrixXd& jacobian) {
            jacobian(0, 0) = 1.0;
        });
    solver_options_lm options(100, 1e-12, 1e-12, 1e-12);
    Eigen::VectorXd parameters(1);
    parameters[0] = 0.0;

    const solver_output result = solver.solve(parameters, options);

    EXPECT_NEAR(parameters[0], 3.0, 1e-6);
    EXPECT_NE(result.status_, solver_convergence_enum::NOT_CONVERGED);
}

TEST(OptimizationAlgorithm, BrentFindsRoot)
{
    double root = 0.0;
    const bool converged = root_finding_algorithms::brent(
        [](double value) { return value * value - 4.0; },
        0.0,
        3.0,
        root,
        0.0,
        1e-12);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(root, 2.0, 1e-10);
}

}  // namespace
}  // namespace solverslib
