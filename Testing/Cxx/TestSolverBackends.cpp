#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "optimization_test_problems.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_nlopt.h"
#include "solver_wrapper.h"
#include "solvers/ceres_solver.h"
#include "solvers/nlopt_solver.h"

namespace solverslib
{
namespace
{
using testing::optimization_test_problem;

double residual_norm(
    const optimization_test_problem& problem, const std::vector<double>& parameters)
{
    vector_type x = to_vector_type(parameters);
    vector_type r = make_vector(problem.num_residuals);
    problem.residuals(x, r);
    return r.norm();
}

void expect_solved(const optimization_test_problem& problem,
    const std::vector<double>&                      parameters,
    double                                          tolerance)
{
    EXPECT_LT(residual_norm(problem, parameters), tolerance) << "problem=" << problem.name;
    for (size_t i = 0; i < problem.expected_solution.size(); ++i)
    {
        EXPECT_NEAR(parameters[i], problem.expected_solution[i], std::sqrt(tolerance))
            << "problem=" << problem.name << " parameter=" << i;
    }
}

std::string problem_name(const ::testing::TestParamInfo<optimization_test_problem>& info)
{
    return info.param.name;
}

// Every backend is exercised through solver_wrapper against the same shared
// problem set (Testing/Cxx/optimization_test_problems.h), so a solver-specific
// fixture only needs to supply the solver_options.
class SolverBackendTest : public ::testing::TestWithParam<optimization_test_problem>
{
};

TEST_P(SolverBackendTest, LevenbergMarquardtSolvesProblem)
{
    const optimization_test_problem& problem = GetParam();
    solver_wrapper                   wrapper(
        problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
    std::shared_ptr<const solver_options> options =
        std::make_shared<solver_options_lm>(500, 1e-14, 1e-14, 1e-14);
    std::vector<double> parameters = problem.initial_guess;

    EXPECT_TRUE(wrapper.solve(parameters, options));
    expect_solved(problem, parameters, 1e-6);
}

TEST_P(SolverBackendTest, LbfgsSolvesProblem)
{
    const optimization_test_problem& problem = GetParam();
    solver_wrapper                   wrapper(
        problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
    std::shared_ptr<const solver_options> options =
        std::make_shared<solver_options_bfgs>(500, 1e-14, 1e-14, 1e-14);
    std::vector<double> parameters = problem.initial_guess;

    EXPECT_TRUE(wrapper.solve(parameters, options));
    expect_solved(problem, parameters, 1e-4);
}

TEST_P(SolverBackendTest, NloptSolvesProblem)
{
    if (!nlopt_solver::is_supported())
    {
        GTEST_SKIP() << "NLopt backend not compiled in (SOLVERS_ENABLE_NLOPT=OFF)";
    }

    const optimization_test_problem& problem = GetParam();
    solver_wrapper                   wrapper(
        problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
    std::shared_ptr<const solver_options> options = std::make_shared<solver_options_nlopt>(
        nlopt_algo_name_enum::LBFGS, 500, 1e-14, 1e-14, 1e-14);
    std::vector<double> parameters = problem.initial_guess;

    EXPECT_TRUE(wrapper.solve(parameters, options));
    expect_solved(problem, parameters, 1e-4);
}

TEST_P(SolverBackendTest, CeresSolvesProblem)
{
    if (!ceres_solver::is_supported())
    {
        GTEST_SKIP() << "Ceres backend not compiled in (SOLVERS_ENABLE_CERES=OFF)";
    }

    const optimization_test_problem& problem = GetParam();
    solver_wrapper                   wrapper(
        problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
    std::shared_ptr<const solver_options> options =
        std::make_shared<solver_options_ceres>(500, 1e-14, 1e-14, 1e-14);
    std::vector<double> parameters = problem.initial_guess;

    EXPECT_TRUE(wrapper.solve(parameters, options));
    expect_solved(problem, parameters, 1e-4);
}

INSTANTIATE_TEST_SUITE_P(AllProblems,
    SolverBackendTest,
    ::testing::Values(testing::make_linear_scalar_problem(),
        testing::make_rosenbrock_problem(),
        testing::make_powell_singular_problem(),
        testing::make_exponential_fit_problem()),
    problem_name);

TEST(SolverBackendSupport, ReportsAllFourSolverTypes)
{
    EXPECT_TRUE(solver_wrapper::is_supported(solver_enum::LM));
    EXPECT_TRUE(solver_wrapper::is_supported(solver_enum::LBFGS));
    EXPECT_EQ(solver_wrapper::is_supported(solver_enum::NLOPT), nlopt_solver::is_supported());
    EXPECT_EQ(solver_wrapper::is_supported(solver_enum::CERES), ceres_solver::is_supported());
}

}  // namespace
}  // namespace solverslib
