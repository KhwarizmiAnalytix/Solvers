#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "detail/eigen_support.h"
#include "optimization_test_problems.h"
#include "solver_options/solver_options_lm.h"
#include "solvers/api/solve.h"
#include "solvers/levenberg_marquardt_solver.h"

namespace solverslib
{
namespace
{
// -- damped_step_solver in isolation --------------------------------------------------
TEST(DampedStepSolver, BothMethodsSolveTheSameDampedSystem)
{
    std::mt19937                     rng(7);
    std::normal_distribution<double> normal;
    const index_type                 m = 30, n = 5;
    matrix_type                      J(m, n);
    vector_type                      y(m);
    for (index_type i = 0; i < m; ++i)
    {
        y[i] = normal(rng);
        for (index_type j = 0; j < n; ++j)
        {
            J(i, j) = normal(rng);
        }
    }
    const matrix_type jtj = J.transpose() * J;
    const vector_type jty = J.transpose() * y;
    vector_type       damping(n);
    damping << 1e-3, 0.5, 2.0, 1e-6, 10.0;

    vector_type reference = (jtj + matrix_type(damping.asDiagonal())).ldlt().solve(jty);
    for (const auto method : {damped_step_method::normal_ldlt, damped_step_method::augmented_qr})
    {
        damped_step_solver solver(m, n, method);
        ASSERT_TRUE(solver.factor(J, jtj, damping));
        vector_type delta(n);
        solver.solve(y, jty, delta);
        EXPECT_LT((delta - reference).norm(), 1e-10 * (1.0 + reference.norm()))
            << "method " << static_cast<int>(method);
    }
}

TEST(DampedStepSolver, QrDoesNotSquareTheConditionNumber)
{
    // Columns with scales 1 and 1e-9: cond(J) = 1e9, cond(J^T J) = 1e18, beyond
    // double precision. The QR form still resolves the small direction.
    const index_type                 m = 20, n = 2;
    matrix_type                      J(m, n);
    vector_type                      y(m);
    std::mt19937                     rng(3);
    std::normal_distribution<double> normal;
    for (index_type i = 0; i < m; ++i)
    {
        J(i, 0) = normal(rng);
        J(i, 1) = 1e-9 * normal(rng);
    }
    const vector_type truth   = (vector_type(2) << 2.0, 3e9).finished();
    y                         = J * truth;
    const matrix_type jtj     = J.transpose() * J;
    const vector_type jty     = J.transpose() * y;
    const vector_type damping = vector_type::Zero(n);  // pure least squares

    damped_step_solver qr(m, n, damped_step_method::augmented_qr);
    ASSERT_TRUE(qr.factor(J, jtj, damping));
    vector_type delta(n);
    qr.solve(y, jty, delta);
    EXPECT_NEAR(delta[0], truth[0], 1e-6);
    EXPECT_NEAR(delta[1] / truth[1], 1.0, 1e-6);
}

TEST(DampedStepSolver, FailuresAreReportedNotThrown)
{
    matrix_type J = matrix_type::Zero(4, 2);
    J.col(0).setOnes();  // second column is zero: rank deficient without damping
    const matrix_type jtj     = J.transpose() * J;
    const vector_type damping = vector_type::Zero(2);

    damped_step_solver qr(4, 2, damped_step_method::augmented_qr);
    EXPECT_FALSE(qr.factor(J, jtj, damping)) << "rank-deficient QR must ask for more damping";

    matrix_type nan_jtj = jtj;
    nan_jtj(0, 0)       = std::numeric_limits<double>::quiet_NaN();
    damped_step_solver ldlt(4, 2, damped_step_method::normal_ldlt);
    EXPECT_FALSE(ldlt.factor(J, nan_jtj, damping));

    // Damping restores a valid factorization in both forms.
    const vector_type positive = vector_type::Constant(2, 1e-3);
    EXPECT_TRUE(qr.factor(J, jtj, positive));
    EXPECT_TRUE(ldlt.factor(J, jtj, positive));
}

// -- literature problems under both linear solvers --------------------------------------
using LinearSolver = levenberg_marquardt_linear_solver_enum;

class LmLinearSolvers : public ::testing::TestWithParam<LinearSolver>
{
};

TEST_P(LmLinearSolvers, SolveTheStandardLeastSquaresProblems)
{
    for (const auto& tp : testing::make_all_test_problems())
    {
        SCOPED_TRACE(tp.name);
        auto options = solver_options_lm_builder()
                           .with_linear_solver(GetParam())
                           .with_max_iterations(500)
                           .with_function_tolerance(1e-24)
                           .with_gradient_tolerance(1e-14)
                           .with_parameter_tolerance(1e-14)
                           .build();
        levenberg_marquardt_solver solver(
            tp.num_parameters, tp.num_residuals, tp.residuals, tp.jacobian);
        vector_type x      = to_vector_type(tp.initial_guess);
        const auto  result = solver.solve(x, *options);
        EXPECT_TRUE(result.converged()) << "status " << static_cast<int>(result.status);
        for (std::size_t i = 0; i < tp.num_parameters; ++i)
        {
            // Powell's singular function has a singular Jacobian at the solution,
            // so its parameters are only determined to a few digits.
            EXPECT_NEAR(x[static_cast<index_type>(i)], tp.expected_solution[i], 1e-3);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BothSolvers,
    LmLinearSolvers,
    ::testing::Values(LinearSolver::NORMAL_LDLT, LinearSolver::AUGMENTED_QR));

TEST(LmLinearSolvers, ApiOptionSelectsTheSolverAndBothAgree)
{
    const auto                 tp = testing::make_rosenbrock_problem();
    api::least_squares_problem problem;
    problem.num_parameters = tp.num_parameters;
    problem.num_residuals  = tp.num_residuals;
    problem.residuals      = tp.residuals;
    problem.set_jacobian(tp.jacobian);

    api::solve_options options;
    options.backend   = api::backend::native;
    options.algorithm = api::algorithm::levenberg_marquardt;
    options.lm        = api::lm_options{};

    options.lm->linear_solver = api::lm_linear_solver::normal_ldlt;
    const auto ldlt           = api::solve(problem, to_vector_type(tp.initial_guess), options);
    options.lm->linear_solver = api::lm_linear_solver::augmented_qr;
    const auto qr             = api::solve(problem, to_vector_type(tp.initial_guess), options);

    ASSERT_TRUE(ldlt.converged()) << ldlt.message;
    ASSERT_TRUE(qr.converged()) << qr.message;
    EXPECT_LT((ldlt.parameters - qr.parameters).norm(), 1e-6);
    EXPECT_NEAR(qr.parameters[0], 1.0, 1e-6);
}

// -- function tolerance is defined on F = 0.5 ||r||^2 ------------------------------------
TEST(LmFunctionTolerance, StopsWhenHalfSquaredResidualNormIsBelowTolerance)
{
    // r(x) = x - 1 starting at x = 1 + 2e-4 has F = 2e-8. A tolerance of 1e-7 on F
    // is already met (||r|| = 2e-4 would NOT satisfy ||r|| < 1e-7), so the solver
    // must report convergence without taking a step.
    levenberg_marquardt_solver solver(
        1,
        1,
        [](const vector_type& p, vector_type& r) { r[0] = p[0] - 1.0; },
        [](const vector_type&, matrix_type& j) { j(0, 0) = 1.0; });
    auto options = solver_options_lm_builder().with_function_tolerance(1e-7).build();

    vector_type x(1);
    x << 1.0 + 2e-4;
    const auto result = solver.solve(x, *options);
    EXPECT_EQ(result.status, native_convergence::function_converged);
    EXPECT_EQ(result.iterations, 0u);
    EXPECT_DOUBLE_EQ(x[0], 1.0 + 2e-4);
}
}  // namespace
}  // namespace solverslib
