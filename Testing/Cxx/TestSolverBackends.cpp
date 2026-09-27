#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "optimization_test_problems.h"
#include "solvers/api/solve.h"

namespace solverslib
{
namespace
{
using testing::optimization_test_problem;

double residual_norm(const optimization_test_problem& problem, const vector_type& parameters)
{
    vector_type r = make_vector(problem.num_residuals);
    problem.residuals(parameters, r);
    return r.norm();
}

void expect_solved(
    const optimization_test_problem& problem, const api::solver_result& result, double tolerance)
{
    EXPECT_LT(residual_norm(problem, result.parameters), tolerance) << "problem=" << problem.name;
    for (size_t i = 0; i < problem.expected_solution.size(); ++i)
    {
        EXPECT_NEAR(result.parameters[i], problem.expected_solution[i], std::sqrt(tolerance))
            << "problem=" << problem.name << " parameter=" << i;
    }
}

std::string problem_name(const ::testing::TestParamInfo<optimization_test_problem>& info)
{
    return info.param.name;
}

api::least_squares_problem to_ls(const optimization_test_problem& tp)
{
    api::least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;
    p.jacobian       = tp.jacobian;
    return p;
}

api::optimization_problem to_obj(const optimization_test_problem& tp)
{
    api::optimization_problem p;
    p.num_parameters = tp.num_parameters;
    p.objective      = [&tp](const vector_type& x) -> double
    {
        vector_type r = make_vector(tp.num_residuals);
        tp.residuals(x, r);
        return 0.5 * r.squaredNorm();
    };
    p.gradient = [&tp](const vector_type& x, vector_type& g)
    {
        vector_type r = make_vector(tp.num_residuals);
        matrix_type J = make_matrix(tp.num_residuals, tp.num_parameters);
        tp.residuals(x, r);
        tp.jacobian(x, J);
        g = J.transpose() * r;
    };
    return p;
}

class SolverBackendTest : public ::testing::TestWithParam<optimization_test_problem>
{
};

TEST_P(SolverBackendTest, LevenbergMarquardtSolvesProblem)
{
    const auto& tp = GetParam();
    auto        ls = to_ls(tp);
    vector_type x0 = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.algorithm           = api::algorithm::levenberg_marquardt;
    opts.backend             = api::backend::native;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(ls, x0, opts);
    EXPECT_TRUE(result.converged());
    expect_solved(tp, result, 1e-6);
}

TEST_P(SolverBackendTest, LbfgsSolvesProblem)
{
    const auto& tp  = GetParam();
    auto        obj = to_obj(tp);
    vector_type x0  = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.algorithm           = api::algorithm::lbfgs;
    opts.backend             = api::backend::native;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(obj, x0, opts);
    EXPECT_TRUE(result.converged());
    expect_solved(tp, result, 1e-4);
}

TEST_P(SolverBackendTest, GaussNewtonSolvesProblem)
{
    const auto& tp = GetParam();
    auto        ls = to_ls(tp);
    vector_type x0 = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.algorithm           = api::algorithm::gauss_newton;
    opts.backend             = api::backend::native;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(ls, x0, opts);
    EXPECT_TRUE(result.converged());
    expect_solved(tp, result, 1e-6);
}

TEST_P(SolverBackendTest, CeresSolvesProblem)
{
    const auto& tp = GetParam();
    auto        ls = to_ls(tp);
    vector_type x0 = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.backend             = api::backend::ceres;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(ls, x0, opts);
    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "Ceres backend not compiled in (SOLVERS_ENABLE_CERES=OFF)";
    }

    EXPECT_TRUE(result.converged());
    expect_solved(tp, result, 1e-4);
}

TEST_P(SolverBackendTest, IpoptSolvesProblem)
{
    const auto& tp  = GetParam();
    auto        obj = to_obj(tp);
    vector_type x0  = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.backend        = api::backend::ipopt;
    opts.max_iterations = 500;
    opts.ipopt          = api::ipopt_options{.tol = 1e-10};

    auto result = api::solve(obj, x0, opts);
    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "Ipopt backend not compiled in (SOLVERS_ENABLE_IPOPT=OFF)";
    }

    EXPECT_TRUE(result.converged());
    expect_solved(tp, result, 1e-4);
}

TEST_P(SolverBackendTest, PetscTaoSolvesProblem)
{
    const auto& tp  = GetParam();
    auto        obj = to_obj(tp);
    vector_type x0  = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.backend        = api::backend::petsc_tao;
    opts.max_iterations = 500;
    opts.petsc_tao      = api::petsc_tao_options{.gatol = 1e-8, .grtol = 1e-8};

    auto result = api::solve(obj, x0, opts);
    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "PETSc/TAO backend not compiled in (SOLVERS_ENABLE_PETSC=OFF)";
    }

    EXPECT_TRUE(result.has_usable_iterate());
}

TEST_P(SolverBackendTest, PoundersSolvesProblem)
{
    const auto& tp = GetParam();

    api::least_squares_problem ls;
    ls.num_parameters = tp.num_parameters;
    ls.num_residuals  = tp.num_residuals;
    ls.residuals      = tp.residuals;
    // No Jacobian — derivative-free path.

    vector_type x0 = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.algorithm      = api::algorithm::pounders;
    opts.backend        = api::backend::pounders;
    opts.max_iterations = 500;
    opts.petsc_tao      = api::petsc_tao_options{.gatol = 1e-6, .grtol = 1e-6};

    auto result = api::solve(ls, x0, opts);
    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "PETSc/TAO backend not compiled in (SOLVERS_ENABLE_PETSC=OFF)";
    }

    EXPECT_TRUE(result.converged());
}

TEST_P(SolverBackendTest, CeresWithInternalJacobian)
{
    const auto& tp = GetParam();
    auto        ls = to_ls(tp);
    vector_type x0 = to_vector_type(tp.initial_guess);

    // Create a least squares problem WITHOUT providing jacobian
    // Ceres will compute it via finite differences internally
    api::least_squares_problem ls_no_jac;
    ls_no_jac.num_parameters = tp.num_parameters;
    ls_no_jac.num_residuals  = tp.num_residuals;
    ls_no_jac.residuals      = tp.residuals;
    ls_no_jac.jacobian       = nullptr;  // No user-provided jacobian

    api::solve_options opts;
    opts.backend             = api::backend::ceres;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(ls_no_jac, x0, opts);
    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "Ceres backend not compiled in (SOLVERS_ENABLE_CERES=OFF)";
    }

    EXPECT_TRUE(result.converged());
    expect_solved(tp, result, 1e-3);  // Slightly relaxed due to finite differences
}

TEST(SolverBackendTest, CeresWithBoundedLeastSquares)
{
    // Rosenbrock problem: solution at (1, 1)
    // Add bounds to constrain the solution
    const auto&                tp = testing::make_rosenbrock_problem();
    api::least_squares_problem ls;
    ls.num_parameters = tp.num_parameters;
    ls.num_residuals  = tp.num_residuals;
    ls.residuals      = tp.residuals;
    ls.jacobian       = tp.jacobian;

    // Constrain first parameter to [0.5, 1.5] and second to [0.5, 1.5]
    ls.bounds.lower = {0.5, 0.5};
    ls.bounds.upper = {1.5, 1.5};

    vector_type x0 = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.backend             = api::backend::ceres;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(ls, x0, opts);

    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "Ceres backend not compiled in (SOLVERS_ENABLE_CERES=OFF)";
    }

    EXPECT_TRUE(result.converged());
    // Check that solution respects bounds
    EXPECT_GE(result.parameters[0], 0.5 - 1e-6);
    EXPECT_LE(result.parameters[0], 1.5 + 1e-6);
    EXPECT_GE(result.parameters[1], 0.5 - 1e-6);
    EXPECT_LE(result.parameters[1], 1.5 + 1e-6);
    // And is close to the true solution
    EXPECT_NEAR(result.parameters[0], 1.0, 1e-3);
    EXPECT_NEAR(result.parameters[1], 1.0, 1e-3);
}

TEST(SolverBackendTest, CeresWithActiveBounds)
{
    // Rosenbrock problem constrained to force active bounds
    // Solution (1, 1) but constrain to [0, 0.8] and [0, 0.8]
    // Expected solution should hit the bounds
    const auto&                tp = testing::make_rosenbrock_problem();
    api::least_squares_problem ls;
    ls.num_parameters = tp.num_parameters;
    ls.num_residuals  = tp.num_residuals;
    ls.residuals      = tp.residuals;
    ls.jacobian       = tp.jacobian;

    // Tight bounds that force solution near boundary
    ls.bounds.lower = {0.0, 0.0};
    ls.bounds.upper = {0.8, 0.8};

    vector_type x0 = to_vector_type(tp.initial_guess);

    api::solve_options opts;
    opts.backend             = api::backend::ceres;
    opts.max_iterations      = 500;
    opts.function_tolerance  = 1e-14;
    opts.parameter_tolerance = 1e-14;

    auto result = api::solve(ls, x0, opts);

    if (result.status == api::solver_status::backend_unavailable)
    {
        GTEST_SKIP() << "Ceres backend not compiled in (SOLVERS_ENABLE_CERES=OFF)";
    }

    EXPECT_TRUE(result.has_usable_iterate());
    // Check bounds are respected
    EXPECT_GE(result.parameters[0], 0.0 - 1e-6);
    EXPECT_LE(result.parameters[0], 0.8 + 1e-6);
    EXPECT_GE(result.parameters[1], 0.0 - 1e-6);
    EXPECT_LE(result.parameters[1], 0.8 + 1e-6);
}

TEST(AutomaticDifferentiation, APISupportsTemplatedResiduals)
{
    // Test that the API can store and retrieve templated residuals
    api::least_squares_problem ls;
    ls.num_parameters = 2;
    ls.num_residuals  = 2;

    const auto& tp = testing::make_rosenbrock_problem();
    ls.residuals   = tp.residuals;
    ls.jacobian    = tp.jacobian;

    // Set templated residuals
    testing::RosenbrocResiduals templated_func;
    ls.set_templated_residuals(templated_func);

    EXPECT_TRUE(ls.has_templated_residuals());

    // Retrieve and verify
    const auto* retrieved = ls.get_templated_residuals<testing::RosenbrocResiduals>();
    EXPECT_NE(retrieved, nullptr);
}

TEST(AutomaticDifferentiation, TemplatedResidualsMatchNumeric)
{
    // Validate that templated residual functors produce identical results
    // to numeric residuals when instantiated with double
    const double x_vals[] = {1.5, 0.5};
    double       r_templated[2];
    double       r_numeric[2];

    // Test Rosenbrock
    testing::RosenbrocResiduals templated_func;
    const auto&                 numeric_problem = testing::make_rosenbrock_problem();
    vector_type                 x = to_vector_type(std::vector<double>(x_vals, x_vals + 2));
    vector_type                 r_numeric_vec = make_vector(2);

    templated_func(x_vals, r_templated);
    numeric_problem.residuals(x, r_numeric_vec);

    EXPECT_NEAR(r_templated[0], r_numeric_vec[0], 1e-14);
    EXPECT_NEAR(r_templated[1], r_numeric_vec[1], 1e-14);
}

TEST(AutomaticDifferentiation, TemplatedLinearScalarMatchesNumeric)
{
    // Test LinearScalar templated functor
    const double x_val = 1.5;
    double       r_templated;
    double       r_numeric;

    testing::LinearScalarResiduals templated_func;
    const auto&                    numeric_problem = testing::make_linear_scalar_problem();
    vector_type                    x               = to_vector_type(std::vector<double>{x_val});
    vector_type                    r_numeric_vec   = make_vector(1);

    templated_func(&x_val, &r_templated);
    numeric_problem.residuals(x, r_numeric_vec);

    EXPECT_NEAR(r_templated, r_numeric_vec[0], 1e-14);
}

TEST(AutomaticDifferentiation, TemplatedPowellSingularMatchesNumeric)
{
    // Test Powell Singular templated functor
    const double x_vals[] = {3.0, -1.0, 0.0, 1.0};
    double       r_templated[4];
    double       r_numeric[4];

    testing::PowellSingularResiduals templated_func;
    const auto&                      numeric_problem = testing::make_powell_singular_problem();
    vector_type                      x = to_vector_type(std::vector<double>(x_vals, x_vals + 4));
    vector_type                      r_numeric_vec = make_vector(4);

    templated_func(x_vals, r_templated);
    numeric_problem.residuals(x, r_numeric_vec);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(r_templated[i], r_numeric_vec[i], 1e-12) << "Mismatch at residual " << i;
    }
}

TEST(AutomaticDifferentiation, TemplatedExponentialFitMatchesNumeric)
{
    // Test Exponential Fit templated functor
    const double x_vals[] = {1.5, -0.2};

    const auto& numeric_problem = testing::make_exponential_fit_problem();
    vector_type x               = to_vector_type(std::vector<double>(x_vals, x_vals + 2));
    vector_type r_numeric_vec   = make_vector(numeric_problem.num_residuals);

    numeric_problem.residuals(x, r_numeric_vec);

    // Manually create the exponential functor with the same data
    std::vector<double> sample_times(10);
    std::vector<double> sample_values(10);
    for (size_t i = 0; i < 10; ++i)
    {
        sample_times[i]  = static_cast<double>(i);
        sample_values[i] = 2.0 * std::exp(-0.3 * sample_times[i]);
    }

    testing::ExponentialFitResiduals templated_func(sample_times, sample_values);
    std::vector<double>              r_templated(10);

    templated_func(x_vals, r_templated.data());

    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_NEAR(r_templated[i], r_numeric_vec[i], 1e-12) << "Mismatch at residual " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(AllProblems,
    SolverBackendTest,
    ::testing::Values(testing::make_linear_scalar_problem(),
        testing::make_rosenbrock_problem(),
        testing::make_powell_singular_problem(),
        testing::make_exponential_fit_problem()),
    problem_name);

}  // namespace
}  // namespace solverslib
