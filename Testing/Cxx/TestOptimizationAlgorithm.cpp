#include <gtest/gtest.h>

#include "solvers/api/solve.h"
#include "solvers/root_finding_algorithms.h"

namespace solverslib
{
namespace
{

TEST(OptimizationAlgorithm, LevenbergMarquardtSolvesScalarResidual)
{
    api::least_squares_problem ls;
    ls.num_parameters = 1;
    ls.num_residuals  = 1;
    ls.residuals = [](const vector_type& x, vector_type& r) { r(0) = x(0) - 3.0; };
    ls.jacobian  = [](const vector_type&, matrix_type& J) { J(0, 0) = 1.0; };

    api::solve_options opts;
    opts.algorithm           = api::algorithm::levenberg_marquardt;
    opts.backend             = api::backend::native;
    opts.max_iterations      = 100;
    opts.function_tolerance  = 1e-12;
    opts.gradient_tolerance  = 1e-12;
    opts.parameter_tolerance = 1e-12;

    vector_type x0(1);
    x0(0) = 0.0;

    auto result = api::solve(ls, x0, opts);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters(0), 3.0, 1e-6);
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
