#include <gtest/gtest.h>

#include "include/detail/eigen_support.h"
#include "solvers/api/solve.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/nlopt_solver.h"
#include "solvers/petsc_tao_solver.h"

namespace solverslib::api
{
namespace
{
// r(x) = [x0 - 3, x1 + 1]; minimizer (3, -1), zero residual, Jacobian = I.
least_squares_problem make_linear_problem(bool with_jacobian)
{
    least_squares_problem problem;
    problem.num_parameters = 2;
    problem.num_residuals  = 2;
    problem.residuals      = [](const vector_type& x, vector_type& r)
    {
        r.resize(2);
        r[0] = x[0] - 3.0;
        r[1] = x[1] + 1.0;
    };
    if (with_jacobian)
    {
        problem.jacobian = [](const vector_type&, matrix_type& j)
        {
            j = matrix_type::Identity(2, 2);
        };
    }
    return problem;
}

TEST(SolverApiDispatch, TraitsReflectProblem)
{
    const problem_traits traits = inspect(make_linear_problem(true));
    EXPECT_TRUE(traits.is_least_squares);
    EXPECT_TRUE(traits.has_jacobian);
    EXPECT_FALSE(traits.has_bounds);
    EXPECT_EQ(traits.num_parameters, 2u);
    EXPECT_EQ(traits.num_residuals, 2u);
}

TEST(SolverApiDispatch, AutoSelectsLmForJacobianLeastSquares)
{
    const problem_traits traits = inspect(make_linear_problem(true));
    EXPECT_EQ(select_algorithm(traits, {}), algorithm::levenberg_marquardt);
    EXPECT_EQ(select_backend(traits, {}), backend::native);
}

TEST(SolverApiDispatch, AutoSelectsPoundersWithoutJacobian)
{
    const problem_traits traits = inspect(make_linear_problem(false));
    EXPECT_EQ(select_algorithm(traits, {}), algorithm::pounders);
    EXPECT_EQ(select_backend(traits, {}), backend::pounders);
}

TEST(SolverApiDispatch, LargeScaleRoutesToTao)
{
    problem_traits traits = inspect(make_linear_problem(true));
    traits.num_residuals  = 100000;  // exceeds default residual threshold
    EXPECT_TRUE(is_large_scale(traits, {}));
    EXPECT_EQ(select_algorithm(traits, {}), algorithm::newton_krylov);
    EXPECT_EQ(select_backend(traits, {}), backend::petsc_tao);
}

TEST(SolverApiDispatch, NativeLmSolvesLinearLeastSquares)
{
    const auto  problem = make_linear_problem(true);
    vector_type x0      = vector_type::Zero(2);

    const solver_result result = solve(problem, x0);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.backend, backend::native);
    EXPECT_EQ(result.algorithm, algorithm::levenberg_marquardt);
    ASSERT_EQ(result.parameters.size(), 2);
    EXPECT_NEAR(result.parameters[0], 3.0, 1e-6);
    EXPECT_NEAR(result.parameters[1], -1.0, 1e-6);
    // The caller's initial guess is never mutated.
    EXPECT_DOUBLE_EQ(x0[0], 0.0);
}

TEST(SolverApiDispatch, GaussNewtonHonorsExplicitAlgorithm)
{
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.algorithm = algorithm::gauss_newton;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.algorithm, algorithm::gauss_newton);
    EXPECT_EQ(result.backend, backend::native);
    EXPECT_TRUE(result.has_usable_iterate());
}

TEST(SolverApiDispatch, BoundsRejectedRatherThanDropped)
{
    auto problem          = make_linear_problem(true);
    problem.bounds.lower  = {0.0, -10.0};  // one-sided is enough to trigger

    const solver_result result = solve(problem, vector_type::Zero(2));
    EXPECT_EQ(result.status, solver_status::unsupported_capability);
    EXPECT_FALSE(result.has_usable_iterate());
}

TEST(SolverApiDispatch, InvalidProblemRejectedBeforeSolving)
{
    least_squares_problem problem;  // no callback, zero dimensions
    const solver_result   result = solve(problem, vector_type{});
    EXPECT_EQ(result.status, solver_status::invalid_problem);
}

TEST(SolverApiDispatch, MismatchedInitialGuessRejected)
{
    const auto          problem = make_linear_problem(true);
    const solver_result result  = solve(problem, vector_type::Zero(3));
    EXPECT_EQ(result.status, solver_status::invalid_problem);
}

TEST(SolverApiDispatch, UnwiredBackendReportedUnavailable)
{
    // PETSc/TAO has no adapter regardless of build flags, so this is a stable
    // assertion of the backend_unavailable contract.
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.backend = backend::petsc_tao;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.status, solver_status::backend_unavailable);
    EXPECT_EQ(result.backend, backend::petsc_tao);
}

TEST(SolverApiDispatch, OptimizationProblemReportsUnimplementedNativePath)
{
    optimization_problem problem;
    problem.num_parameters = 2;
    problem.objective      = [](const vector_type& x) { return x.squaredNorm(); };

    const solver_result result = solve(problem, vector_type::Zero(2));
    EXPECT_EQ(result.status, solver_status::unsupported_capability);
    EXPECT_EQ(result.backend, backend::native);
    EXPECT_EQ(result.algorithm, algorithm::lbfgs);
}

// -- Ceres backend -----------------------------------------------------------
// These exercise the real adapter when Ceres is compiled in, and assert the
// backend_unavailable contract otherwise (matching the optional-backend skip
// pattern used across the suite).
TEST(SolverApiDispatch, CeresSolvesWhenAvailable)
{
    if (!solverslib::ceres_solver::is_supported())
    {
        GTEST_SKIP() << "Ceres backend not compiled in";
    }
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.backend = backend::ceres;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.backend, backend::ceres);
    EXPECT_TRUE(result.has_usable_iterate());
    ASSERT_TRUE(result.residual_norm.has_value());
    EXPECT_LT(*result.residual_norm, 1e-6);
    EXPECT_NEAR(result.parameters[0], 3.0, 1e-5);
    EXPECT_NEAR(result.parameters[1], -1.0, 1e-5);
}

TEST(SolverApiDispatch, CeresOptionsAreAccepted)
{
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.backend            = backend::ceres;
    options.ceres              = ceres_options{};
    options.ceres->linear_solver = ceres_linear_solver::dense_normal_cholesky;
    options.ceres->num_threads   = 2;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    if (solverslib::ceres_solver::is_supported())
    {
        EXPECT_TRUE(result.has_usable_iterate());
    }
    else
    {
        EXPECT_EQ(result.status, solver_status::backend_unavailable);
    }
    EXPECT_EQ(result.backend, backend::ceres);
}

TEST(SolverApiDispatch, CeresHonorsBounds)
{
    if (!solverslib::ceres_solver::is_supported())
    {
        GTEST_SKIP() << "Ceres backend not compiled in";
    }
    // Unconstrained minimizer is (3, -1); box the first parameter to [0, 1].
    auto problem         = make_linear_problem(true);
    problem.bounds.lower = {0.0, -10.0};
    problem.bounds.upper = {1.0, 10.0};
    solve_options options;
    options.backend = backend::ceres;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.backend, backend::ceres);
    EXPECT_TRUE(result.has_usable_iterate());
    EXPECT_LE(result.parameters[0], 1.0 + 1e-6);
}

// -- NLopt backend -----------------------------------------------------------
TEST(SolverApiDispatch, NloptSolvesWhenAvailable)
{
    if (!solverslib::nlopt_solver::is_supported())
    {
        GTEST_SKIP() << "NLopt backend not compiled in";
    }
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.backend = backend::nlopt;
    options.nlopt   = nlopt_options{};  // default LBFGS (gradient-based)

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.backend, backend::nlopt);
    EXPECT_TRUE(result.has_usable_iterate());
    ASSERT_TRUE(result.residual_norm.has_value());
    EXPECT_LT(*result.residual_norm, 1e-4);
}

TEST(SolverApiDispatch, NloptGradientAlgorithmNeedsJacobian)
{
    if (!solverslib::nlopt_solver::is_supported())
    {
        GTEST_SKIP() << "NLopt backend not compiled in";
    }
    const auto    problem = make_linear_problem(false);  // no Jacobian
    solve_options options;
    options.backend = backend::nlopt;
    options.nlopt   = nlopt_options{};  // LBFGS requires a gradient

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

TEST(SolverApiDispatch, NloptUnavailableReportsBackendUnavailable)
{
    if (solverslib::nlopt_solver::is_supported())
    {
        GTEST_SKIP() << "NLopt is compiled in; unavailable path not exercised";
    }
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.backend = backend::nlopt;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.status, solver_status::backend_unavailable);
    EXPECT_EQ(result.backend, backend::nlopt);
}

// -- General optimization problem, and its routing --------------------------
// f(x) = ||x - c||^2, gradient 2(x - c); minimizer c.
optimization_problem make_quadratic_problem(const vector_type& c, bool with_gradient)
{
    optimization_problem problem;
    problem.num_parameters = static_cast<std::size_t>(c.size());
    problem.objective      = [c](const vector_type& x) { return (x - c).squaredNorm(); };
    if (with_gradient)
    {
        problem.gradient = [c](const vector_type& x, vector_type& g) { g = 2.0 * (x - c); };
    }
    return problem;
}

TEST(SolverApiDispatch, OptimizationConstraintsRouteToIpopt)
{
    auto problem                = make_quadratic_problem(vector_type::Ones(2), true);
    problem.constraints.num_inequality = 1;
    const problem_traits traits = inspect(problem);
    EXPECT_TRUE(traits.has_nonlinear_constraints);
    EXPECT_EQ(select_algorithm(traits, {}), algorithm::interior_point);
    EXPECT_EQ(select_backend(traits, {}), backend::ipopt);
}

TEST(SolverApiDispatch, OptimizationHessianVectorRoutesToTao)
{
    auto problem = make_quadratic_problem(vector_type::Ones(2), true);
    problem.hessian_vector =
        [](const vector_type&, const vector_type& v, vector_type& out) { out = 2.0 * v; };
    const problem_traits traits = inspect(problem);
    EXPECT_TRUE(traits.has_hessian_vector_product);
    EXPECT_TRUE(is_large_scale(traits, {}));
    EXPECT_EQ(select_algorithm(traits, {}), algorithm::newton_krylov);
    EXPECT_EQ(select_backend(traits, {}), backend::petsc_tao);
}

// -- POUNDERS (via PETSc/TAO) ------------------------------------------------
TEST(SolverApiDispatch, PoundersRouteUsesTaoAdapter)
{
    const auto    problem = make_linear_problem(false);  // no Jacobian -> pounders
    solve_options options;  // automatic

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.backend, backend::pounders);
    if (solverslib::petsc_tao_solver::is_supported())
    {
        EXPECT_TRUE(result.has_usable_iterate());
        ASSERT_TRUE(result.residual_norm.has_value());
        EXPECT_LT(*result.residual_norm, 1e-4);
    }
    else
    {
        EXPECT_EQ(result.status, solver_status::backend_unavailable);
    }
}

TEST(SolverApiDispatch, PetscTaoOptionsAreAccepted)
{
    const auto    problem = make_linear_problem(true);
    solve_options options;
    options.backend                = backend::petsc_tao;
    options.petsc_tao              = petsc_tao_options{};
    options.petsc_tao->algorithm   = tao_algorithm::brgn;
    options.petsc_tao->matrix_free = true;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.backend, backend::petsc_tao);
    if (!solverslib::petsc_tao_solver::is_supported())
    {
        EXPECT_EQ(result.status, solver_status::backend_unavailable);
    }
}

// -- Ipopt -------------------------------------------------------------------
TEST(SolverApiDispatch, IpoptSolvesWhenAvailableOtherwiseUnavailable)
{
    const auto    problem = make_quadratic_problem(vector_type::Constant(2, 2.0), true);
    solve_options options;
    options.backend = backend::ipopt;
    options.ipopt   = ipopt_options{};

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.backend, backend::ipopt);
    if (solverslib::ipopt_solver::is_supported())
    {
        EXPECT_TRUE(result.has_usable_iterate());
        EXPECT_NEAR(result.parameters[0], 2.0, 1e-5);
    }
    else
    {
        EXPECT_EQ(result.status, solver_status::backend_unavailable);
    }
}

TEST(SolverApiDispatch, IpoptRequiresGradientWhenAvailable)
{
    if (!solverslib::ipopt_solver::is_supported())
    {
        GTEST_SKIP() << "Ipopt not compiled in; gradient-required path not exercised";
    }
    const auto    problem = make_quadratic_problem(vector_type::Ones(2), false);  // no gradient
    solve_options options;
    options.backend = backend::ipopt;

    const solver_result result = solve(problem, vector_type::Zero(2), options);
    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

TEST(SolverApiDispatch, NativeObjectivePathStillUnsupported)
{
    // Small unconstrained objective auto-routes to native L-BFGS, which has no
    // scalar-objective kernel yet.
    const auto    problem = make_quadratic_problem(vector_type::Ones(2), true);
    const problem_traits traits = inspect(problem);
    EXPECT_EQ(select_backend(traits, {}), backend::native);

    const solver_result result = solve(problem, vector_type::Zero(2));
    EXPECT_EQ(result.status, solver_status::unsupported_capability);
    EXPECT_EQ(result.backend, backend::native);
}
}  // namespace
}  // namespace solverslib::api
