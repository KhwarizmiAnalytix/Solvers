#include <gtest/gtest.h>

#include "solvers/api/solve.h"

namespace solverslib::api
{
namespace
{

TEST(SolveApi, SolvesSimpleLeastSquares)
{
    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 1;
    problem.residuals      = [](const vector_type& x, vector_type& r) { r[0] = x[0] - 2.0; };
    problem.set_jacobian([](const vector_type&, matrix_type& J) { J(0, 0) = 1.0; });

    vector_type x0(1);
    x0 << 0.0;

    auto result = solve(problem, x0);

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters[0], 2.0, 1e-6);
    EXPECT_EQ(result.backend, backend::native);
}

TEST(SolveApi, ReturnsBackendUnavailableForMissingBackend)
{
    least_squares_problem problem;
    problem.num_parameters = 1;
    problem.num_residuals  = 1;
    problem.residuals      = [](const vector_type& x, vector_type& r) { r[0] = x[0] - 2.0; };

    vector_type x0(1);
    x0 << 0.0;

    solve_options opts;
    opts.algorithm = algorithm::pounders;
    opts.backend   = backend::pounders;

    auto result = solve(problem, x0, opts);

    // Either converges (PETSc compiled in) or reports unavailable.
    EXPECT_TRUE(result.converged() || result.status == solver_status::backend_unavailable);
}

}  // namespace
}  // namespace solverslib::api
