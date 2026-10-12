#include <gtest/gtest.h>

#include <cmath>

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_nlopt.h"
#include "solver_options/solver_options_petsc.h"
#include "solver_options/solver_options_rnc_lm.h"

#if SOLVERS_HAS_CERES
#include "solvers/ceres_solver.h"
#endif
#if SOLVERS_HAS_IPOPT
#include "solvers/ipopt_solver.h"
#endif
#if SOLVERS_HAS_NLOPT
#include "solvers/nlopt_solver.h"
#endif
#if SOLVERS_HAS_PETSC
#include "solvers/petsc_tao_solver.h"
#endif

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::test;

namespace
{
constexpr double kMinimumTolerance = 1e-4;

#define REQUIRE_BACKEND(available, name)                                                           \
    if (!(available))                                                                              \
    {                                                                                              \
        GTEST_SKIP() << (name) << " backend not compiled in";                                      \
    }

void expect_rosenbrock_minimum(const solver_result& result)
{
    ASSERT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters(0), 1.0, kMinimumTolerance);
    EXPECT_NEAR(result.parameters(1), 1.0, kMinimumTolerance);
    EXPECT_NEAR(result.objective, 0.0, kMinimumTolerance);
}

// Exact Hessian of f(x) = 0.5 * ||r(x)||^2 for the Rosenbrock residuals used
// above (r1 = 10(x2 - x1^2), r2 = 1 - x1). Exercises the optimization-path
// hessian callback that Ipopt/PETSc only call when an exact Hessian is
// requested; rosenbrock_optimization() never sets one.
void rosenbrock_hessian(const vector_type& x, matrix_type& h)
{
    h.resize(2, 2);
    h(0, 0) = 600.0 * x(0) * x(0) - 200.0 * x(1) + 1.0;
    h(0, 1) = -200.0 * x(0);
    h(1, 0) = -200.0 * x(0);
    h(1, 1) = 100.0;
}

optimization_problem rosenbrock_optimization_with_hessian()
{
    optimization_problem problem;
    problem.num_parameters = 2;
    problem.objective      = rosenbrock_objective;
    problem.gradient       = rosenbrock_gradient;
    problem.hessian        = rosenbrock_hessian;
    return problem;
}
}  // namespace

TEST(NativeBackend, Solve)
{
    {
        SCOPED_TRACE("LevenbergMarquardtLeastSquaresWithJacobian");
        const auto options = solver_options_lm_builder()
                                 .with_max_iterations(200)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-14)
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::native);
        EXPECT_EQ(result.algorithm, algorithm::levenberg_marquardt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LevenbergMarquardtLeastSquaresFiniteDifferences");
        const auto options = solver_options_lm_builder()
                                 .with_max_iterations(400)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-14)
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LevenbergMarquardtAllOptionsSet");
        const auto options =
            solver_options_lm_builder()
                .with_max_iterations(1000)
                .with_function_tolerance(1e-13)
                .with_gradient_tolerance(1e-11)
                .with_parameter_tolerance(1e-13)
                .with_verbose(true)
                .with_bold_acceptance(true)
                .with_bold_acceptance_exponent(1.5)
                .with_geodesic_acceleration(false)
                .with_geodesic_acceleration_threshold(0.5)
                .with_geodesic_acceleration_step(0.02)
                .with_initial_damping(1e-3)
                .with_initial_rejection_multiplier(3.0)
                .with_damping_decrease_factor(7.0)
                .with_damping_increase_factor(13.0)
                .with_damping_floor(1e-6)
                .with_nielsen_damping_floor(1e-14)
                .with_damping_ceiling(1e10)
                .with_levenberg_marquardt_damping_ceiling(1e6)
                .with_diagonal_scaling_floor(1e-11)
                .with_roundoff_noise_factor(10.0)
                .with_finite_difference_step(2e-5)
                .with_difference_scale(finite_difference_scale::relative)
                .with_type(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT)
                .with_linear_solver(levenberg_marquardt_linear_solver_enum::AUGMENTED_QR)
                .with_log_file("lm_all_options.log")
                .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::levenberg_marquardt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("GaussNewtonLeastSquares");
        const auto options = solver_options_gn_builder()
                                 .with_max_iterations(200)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-14)
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::gauss_newton);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("GaussNewtonLeastSquaresFiniteDifferences");
        const auto options = solver_options_gn_builder()
                                 .with_max_iterations(400)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-14)
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::gauss_newton);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("GaussNewtonAllOptionsSet");
        const auto options = solver_options_gn_builder()
                                 .with_max_iterations(400)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_parameter_tolerance(1e-13)
                                 .with_verbose(true)
                                 .with_bump(5e-5)
                                 .with_difference_scale(finite_difference_scale::relative)
                                 .with_max_line_search_iterations(30)
                                 .with_line_search_backtracking_factor(0.6)
                                 .with_line_search_sufficient_decrease(1e-3)
                                 .with_log_file("gn_all_options.log")
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::gauss_newton);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSOptimization");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(1000)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-10)
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSAllOptionsSet");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_parameter_tolerance(1e-13)
                                 .with_verbose(true)
                                 .with_method_type(lbfgs_line_search_method_type::WOLFE)
                                 .with_type(lbfgs_line_search_type::BACKTRACKING)
                                 .with_tau(8)
                                 .with_max_iteration_linesearch(60)
                                 .with_step_min(1e-8)
                                 .with_step_max(500000.0)
                                 .with_linesearch_tolerance(1e-10)
                                 .with_linesearch_wolfe(0.8)
                                 .with_initial_step(0.3)
                                 .with_backtracking_decrease(0.6)
                                 .with_backtracking_increase(1.8)
                                 .with_line_search_expansion(4.0)
                                 .with_bump(1e-5)
                                 .with_difference_scale(finite_difference_scale::relative)
                                 .with_log_file("bfgs_all_options.log")
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSConvergesOnParameterStep");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(5000)
                                 .with_function_tolerance(0.0)
                                 .with_gradient_tolerance(0.0)
                                 .with_parameter_tolerance(1e-10)
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        EXPECT_EQ(result.convergence, convergence_reason::parameter);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSConvergesOnGradientNorm");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(5000)
                                 .with_function_tolerance(0.0)
                                 .with_gradient_tolerance(1e-9)
                                 .with_parameter_tolerance(0.0)
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        EXPECT_EQ(result.convergence, convergence_reason::gradient);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSStopsAtMaxIterations");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(1)
                                 .with_function_tolerance(0.0)
                                 .with_gradient_tolerance(0.0)
                                 .with_parameter_tolerance(0.0)
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        EXPECT_EQ(result.status, solver_status::max_iterations);
    }
    {
        SCOPED_TRACE("LBFGSFiniteDifferenceGradient");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-10)
                                 .with_bump(1e-6)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSNocedalWrightTightBracket");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                 .with_initial_step(10.0)
                                 .with_step_min(1e-3)
                                 .with_step_max(20.0)
                                 .with_line_search_expansion(1.05)
                                 .with_linesearch_tolerance(1e-4)
                                 .with_linesearch_wolfe(0.1)
                                 .with_max_iteration_linesearch(200)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSBacktrackingArmijo");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                 .with_type(lbfgs_line_search_type::BACKTRACKING)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSBacktrackingStrongWolfe");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_method_type(lbfgs_line_search_method_type::STRONG_WOLFE)
                                 .with_type(lbfgs_line_search_type::BACKTRACKING)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSBracketingArmijo");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                 .with_type(lbfgs_line_search_type::BRACKETING)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSBracketingWolfe");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_method_type(lbfgs_line_search_method_type::WOLFE)
                                 .with_type(lbfgs_line_search_type::BRACKETING)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSBracketingStrongWolfe");
        const auto options = solver_options_bfgs_builder()
                                 .with_max_iterations(2000)
                                 .with_function_tolerance(1e-13)
                                 .with_gradient_tolerance(1e-11)
                                 .with_method_type(lbfgs_line_search_method_type::STRONG_WOLFE)
                                 .with_type(lbfgs_line_search_type::BRACKETING)
                                 .build();
        const auto result   = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.algorithm, algorithm::lbfgs);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LBFGSLeastSquaresIsUnsupported");
        const auto options = solver_options_bfgs_builder().build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
    {
        SCOPED_TRACE("LevenbergMarquardtOptimizationIsNotRouted");
        const auto options = solver_options_lm_builder().build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::backend_unavailable);
    }
    {
        SCOPED_TRACE("RejectsWrongInitialGuessSize");
        const auto  options = solver_options_lm_builder().build();
        vector_type bad(3);
        bad << 0.0, 0.0, 0.0;
        const auto result = solve(rosenbrock_least_squares(true), bad, *options);

        EXPECT_EQ(result.status, solver_status::invalid_problem);
    }
    {
        SCOPED_TRACE("RNC_LM_WithoutCurveDerivatives_Fails");
        const auto options = solver_options_rnc_lm_builder().build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
}

TEST(CeresBackend, Solve)
{
#if SOLVERS_HAS_CERES
    REQUIRE_BACKEND(ceres_solver::is_supported(), "Ceres");
#else
    GTEST_SKIP() << "Ceres backend not compiled in";
#endif
    {
        SCOPED_TRACE("LeastSquares");
        const auto options = solver_options_ceres_builder().with_max_iterations(200).build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ceres);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LeastSquaresAllOptionsSet");
        // dense_linear_algebra_library_type is intentionally left at its default
        // (EIGEN): the alternatives (LAPACK, CUDA) require optional Ceres build
        // configs not guaranteed to be present, so exercising them here would
        // make this test flaky on machines where Ceres was built without them.
        const auto options =
            solver_options_ceres_builder()
                .with_max_iterations(300)
                .with_function_tolerance(1e-9)
                .with_gradient_tolerance(1e-9)
                .with_parameter_tolerance(1e-9)
                .with_verbose(true)
                .with_log_file("ceres_all_options.log")
                .with_linear_solver_type(linear_solver_enum::DENSE_QR)
                .with_preconditioner_type(preconditioner_enum::IDENTITY)
                .with_minimizer_type(minimizer_enum::LINE_SEARCH)
                .with_trust_region_strategy_type(trust_region_strategy_enum::DOGLEG)
                .with_num_threads(2)
                .with_max_solver_time_in_seconds(30.0)
                .with_dogleg_type(dogleg_enum::SUBSPACE_DOGLEG)
                // (L)BFGS directions require WOLFE line search in Ceres, so
                // line_search_type stays at its default (WOLFE) here.
                .with_line_search_direction_type(line_search_direction_enum::BFGS)
                .with_line_search_type(line_search_enum::WOLFE)
                .with_nonlinear_conjugate_gradient_type(
                    nonlinear_conjugate_gradient_enum::POLAK_RIBIERE)
                .with_line_search_interpolation_type(line_search_interpolation_enum::QUADRATIC)
                .with_numeric_diff_method(numeric_diff_method_enum::FORWARD)
                .with_sparse_linear_algebra_library_type(
                    sparse_linear_algebra_library_enum::NO_SPARSE)
                .with_linear_solver_ordering_type(linear_solver_ordering_enum::NESDIS)
                .with_visibility_clustering_type(visibility_clustering_enum::SINGLE_LINKAGE)
                .with_initial_trust_region_radius(0.2)
                .with_max_trust_region_radius(1e14)
                .with_min_trust_region_radius(1e-30)
                .with_min_relative_decrease(1e-4)
                .with_eta(0.001)
                .with_use_inner_iterations(true)
                .with_inner_iteration_tolerance(1e-12)
                .with_jacobi_scaling(false)
                .with_dynamic_sparsity(true)
                .with_use_mixed_precision_solves(true)
                .with_max_num_refinement_iterations(2)
                .with_max_lbfgs_rank(10)
                .with_min_line_search_step_size(1e-8)
                .with_line_search_sufficient_function_decrease(1e-3)
                .with_line_search_sufficient_curvature_decrease(0.8)
                .with_max_line_search_step_contraction(1e-2)
                .with_min_line_search_step_contraction(0.5)
                .with_max_line_search_step_expansion(8.0)
                .with_gradient_check_relative_precision(1e-7)
                .with_gradient_check_numeric_derivative_relative_step_size(1e-5)
                .with_min_linear_solver_iterations(1)
                .with_max_linear_solver_iterations(300)
                .with_trust_region_minimizer_iterations_to_dump({0})
                .with_trust_region_problem_dump_directory("ceres_all_options_dump")
                .with_trust_region_problem_dump_format_type(dump_format_enum::CONSOLE)
                .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ceres);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationIsUnsupported");
        const auto options = solver_options_ceres_builder().build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
    {
        // No Jacobian callback and no bounds: exercises ceres_solver's
        // auto-generated finite-difference Jacobian along its unconstrained,
        // central-difference path (ceres_solver.cpp central-difference branch).
        SCOPED_TRACE("LeastSquaresFiniteDifferenceNoBounds");
        const auto options = solver_options_ceres_builder().with_max_iterations(300).build();
        const auto result  = solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ceres);
        expect_rosenbrock_minimum(result);
    }
    {
        // Bounds placed exactly on the starting point drive the auto-generated
        // finite-difference Jacobian into its degenerate (step <= 0) branches
        // on both the lower-bound and upper-bound sides. The zeroed Jacobian
        // column pins that parameter at the boundary, so the true minimum is
        // not reachable from here; this is a smoke test for the degenerate
        // code paths themselves (ceres_solver.cpp), not for convergence.
        SCOPED_TRACE("LeastSquaresBoundedFiniteDifferenceDegenerateSteps");
        auto problem         = rosenbrock_least_squares(false);
        problem.bounds.lower = {-1.2, -1e9};
        problem.bounds.upper = {1e9, 1.0};
        const auto options   = solver_options_ceres_builder().with_max_iterations(500).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ceres);
        EXPECT_GE(result.parameters(0), problem.bounds.lower[0]);
        EXPECT_LE(result.parameters(1), problem.bounds.upper[1]);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    {
        // Bounds placed just inside the representable step from the starting
        // point drive the finite-difference Jacobian into its non-degenerate
        // one-sided (forward/backward) branches on both sides. The one-sided
        // steps are extremely small, so the solver can declare premature
        // (parameter-tolerance) convergence away from the true minimum; this
        // exercises the near-boundary branches themselves rather than final
        // accuracy.
        SCOPED_TRACE("LeastSquaresBoundedFiniteDifferenceNearBoundarySteps");
        auto problem         = rosenbrock_least_squares(false);
        problem.bounds.lower = {-1.2 - 5e-9, -1e9};
        problem.bounds.upper = {1e9, 1.0 + 5e-9};
        const auto options   = solver_options_ceres_builder().with_max_iterations(500).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ceres);
        EXPECT_GE(result.parameters(0), problem.bounds.lower[0]);
        EXPECT_LE(result.parameters(1), problem.bounds.upper[1]);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    {
        // Bounds combined with an analytic Jacobian: exercises
        // SetParameterLowerBound/UpperBound independently of the
        // finite-difference path.
        SCOPED_TRACE("LeastSquaresBoundedWithJacobian");
        auto problem         = rosenbrock_least_squares(true);
        problem.bounds.lower = {-5.0, -5.0};
        problem.bounds.upper = {5.0, 5.0};
        const auto options   = solver_options_ceres_builder().with_max_iterations(300).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ceres);
        expect_rosenbrock_minimum(result);
    }
}

TEST(IpoptBackend, Solve)
{
#if SOLVERS_HAS_IPOPT
    REQUIRE_BACKEND(ipopt_solver::is_supported(), "Ipopt");
#else
    GTEST_SKIP() << "Ipopt backend not compiled in";
#endif
    {
        SCOPED_TRACE("LeastSquaresWithJacobian");
        const auto options = solver_options_ipopt_builder().with_max_iterations(500).build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LeastSquaresAllOptionsSet");
        const auto options =
            solver_options_ipopt_builder()
                .with_max_iterations(600)
                .with_function_tolerance(1e-11)
                .with_gradient_tolerance(1e-11)
                .with_parameter_tolerance(1e-11)
                .with_verbose(true)
                .with_tol(1e-9)
                .with_acceptable_tol(1e-7)
                .with_hessian_approximation(ipopt_hessian_approximation_enum::EXACT)
                .with_linear_solver("mumps")
                .with_max_wall_time_seconds(60.0)
                .with_log_file("ipopt_all_options.log")
                .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LeastSquaresWithoutJacobianIsUnsupported");
        const auto options = solver_options_ipopt_builder().build();
        const auto result  = solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
    {
        SCOPED_TRACE("OptimizationWithGradient");
        const auto options = solver_options_ipopt_builder().with_max_iterations(500).build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationWithoutGradientIsUnsupported");
        const auto options = solver_options_ipopt_builder().build();
        const auto result  = solve(rosenbrock_optimization(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
    {
        // Bounds that do not exclude the true minimum: exercises
        // solvers_tnlp::get_bounds_info's non-empty lower_/upper_ branches for
        // a least-squares problem.
        SCOPED_TRACE("LeastSquaresWithBounds");
        auto problem         = rosenbrock_least_squares(true);
        problem.bounds.lower = {-10.0, -10.0};
        problem.bounds.upper = {10.0, 10.0};
        const auto options   = solver_options_ipopt_builder().with_max_iterations(500).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        expect_rosenbrock_minimum(result);
    }
    {
        // Same bound coverage for the scalar optimization path.
        SCOPED_TRACE("OptimizationWithBounds");
        auto problem         = rosenbrock_optimization(true);
        problem.bounds.lower = {-10.0, -10.0};
        problem.bounds.upper = {10.0, 10.0};
        const auto options   = solver_options_ipopt_builder().with_max_iterations(500).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        expect_rosenbrock_minimum(result);
    }
    {
        // use_exact_hessian() true with a real Hessian callback: exercises
        // solvers_tnlp::eval_h's sparsity-query and value-fill branches, which
        // the limited-memory default never reaches.
        SCOPED_TRACE("OptimizationWithExactHessian");
        const auto problem  = rosenbrock_optimization_with_hessian();
        const auto options  = solver_options_ipopt_builder()
                                 .with_max_iterations(200)
                                 .with_hessian_approximation(ipopt_hessian_approximation_enum::EXACT)
                                 .build();
        const auto result   = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        expect_rosenbrock_minimum(result);
    }
    {
        // A one-iteration budget forces Ipopt's Maximum_Iterations_Exceeded
        // return status, exercising the budget_exhausted branch of classify().
        SCOPED_TRACE("StopsAtMaxIterationsIsBudgetExhausted");
        const auto options = solver_options_ipopt_builder().with_max_iterations(1).build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::ipopt);
        EXPECT_EQ(result.status, solver_status::max_iterations);
    }
    // Note: an unrecognized linear_solver name does not reach
    // ipopt_solver::solve's app->Initialize()-failure branch: Ipopt's
    // SetStringValue() rejects the bad value and leaves the default solver in
    // place (its return value is not checked by the caller), so Initialize()
    // still succeeds. That early-return branch (ipopt_solver.cpp, "Ipopt
    // initialization failed") was not found to be reachable from the public
    // API within this test file; see the final report for details.
}

TEST(NloptBackend, Solve)
{
#if SOLVERS_HAS_NLOPT
    REQUIRE_BACKEND(nlopt_solver::is_supported(), "NLopt");
#else
    GTEST_SKIP() << "NLopt backend not compiled in";
#endif
    {
        SCOPED_TRACE("LeastSquaresWithJacobian");
        const auto options = solver_options_nlopt_builder().with_max_iterations(500).build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LeastSquaresWithoutJacobianDerivativeFree");
        const auto options = solver_options_nlopt_builder()
                                 .with_max_iterations(5000)
                                 .with_algorithm(nlopt_algorithm_enum::LN_NELDERMEAD)
                                 .with_xtol_rel(1e-12)
                                 .with_ftol_rel(1e-12)
                                 .build();
        const auto result = solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationWithGradient");
        const auto options = solver_options_nlopt_builder().with_max_iterations(500).build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationAllOptionsSet");
        const auto options = solver_options_nlopt_builder()
                                 .with_max_iterations(800)
                                 .with_function_tolerance(1e-12)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-12)
                                 .with_verbose(true)
                                 .with_algorithm(nlopt_algorithm_enum::LD_SLSQP)
                                 .with_xtol_rel(1e-10)
                                 .with_ftol_rel(1e-10)
                                 .with_max_time(60.0)
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationWithoutGradientIsUnsupported");
        const auto options = solver_options_nlopt_builder().build();
        const auto result  = solve(rosenbrock_optimization(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
    {
        // Exercises the LD_MMA branch of the algorithm switch. MMA is
        // designed around (box-)constrained conservative convex
        // approximations; on this unconstrained problem it is not expected to
        // reach the tight tolerance expect_rosenbrock_minimum requires, so
        // this only checks that the algorithm runs to a finite, usable
        // result (i.e. the switch branch and solve path execute cleanly).
        SCOPED_TRACE("LeastSquaresAlgorithmLD_MMA");
        const auto options = solver_options_nlopt_builder()
                                  .with_max_iterations(2000)
                                  .with_algorithm(nlopt_algorithm_enum::LD_MMA)
                                  .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    {
        // Exercises the LD_CCSAQ branch of the algorithm switch. Same caveat
        // as LD_MMA above: CCSAQ is a conservative convex approximation
        // method, so only a clean run is asserted here.
        SCOPED_TRACE("OptimizationAlgorithmLD_CCSAQ");
        const auto options = solver_options_nlopt_builder()
                                  .with_max_iterations(2000)
                                  .with_algorithm(nlopt_algorithm_enum::LD_CCSAQ)
                                  .build();
        const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        EXPECT_TRUE(std::isfinite(result.objective));
    }
    {
        // Exercises the LN_SBPLX branch of the algorithm switch on a
        // derivative-free least-squares problem (NLopt optimization always
        // requires a gradient callback per dispatch.cpp, so a gradient-free
        // algorithm can only be exercised through the least-squares path).
        SCOPED_TRACE("LeastSquaresAlgorithmLN_SBPLX");
        const auto options = solver_options_nlopt_builder()
                                  .with_max_iterations(20000)
                                  .with_algorithm(nlopt_algorithm_enum::LN_SBPLX)
                                  .with_xtol_rel(1e-12)
                                  .with_ftol_rel(1e-12)
                                  .build();
        const auto result = solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        // Bounds that do not exclude the true minimum: exercises
        // nlopt_set_lower_bounds/upper_bounds for a least-squares problem.
        SCOPED_TRACE("LeastSquaresWithBounds");
        auto problem         = rosenbrock_least_squares(true);
        problem.bounds.lower = {-10.0, -10.0};
        problem.bounds.upper = {10.0, 10.0};
        const auto options   = solver_options_nlopt_builder().with_max_iterations(1000).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        // Same bound coverage for the scalar optimization path.
        SCOPED_TRACE("OptimizationWithBounds");
        auto problem         = rosenbrock_optimization(true);
        problem.bounds.lower = {-10.0, -10.0};
        problem.bounds.upper = {10.0, 10.0};
        const auto options   = solver_options_nlopt_builder().with_max_iterations(1000).build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        expect_rosenbrock_minimum(result);
    }
    {
        // A one-evaluation budget forces NLOPT_MAXEVAL_REACHED, exercising the
        // budget_exhausted branch for exhausted evaluations.
        SCOPED_TRACE("StopsAtMaxEvalIsBudgetExhausted");
        const auto options = solver_options_nlopt_builder().with_max_iterations(1).build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        EXPECT_EQ(result.status, solver_status::max_iterations);
    }
    {
        // An effectively zero time budget forces NLOPT_MAXTIME_REACHED,
        // exercising the budget_exhausted branch for exhausted wall time.
        SCOPED_TRACE("StopsAtMaxTimeIsBudgetExhausted");
        const auto options = solver_options_nlopt_builder()
                                  .with_max_iterations(100000)
                                  .with_max_time(1e-12)
                                  .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::nlopt);
        EXPECT_EQ(result.status, solver_status::max_iterations);
    }
}

TEST(PetscBackend, Solve)
{
#if SOLVERS_HAS_PETSC
    REQUIRE_BACKEND(petsc_tao_solver::is_supported(), "PETSc/TAO");
#else
    GTEST_SKIP() << "PETSc/TAO backend not compiled in";
#endif
    {
        SCOPED_TRACE("LeastSquaresBRGN");
        const auto options = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::BRGN)
                                 .with_max_iterations(200)
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("LeastSquaresAllOptionsSet");
        const auto options = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::BRGN)
                                 .with_max_iterations(400)
                                 .with_function_tolerance(1e-11)
                                 .with_gradient_tolerance(1e-11)
                                 .with_parameter_tolerance(1e-11)
                                 .with_verbose(true)
                                 .with_gatol(1e-9)
                                 .with_grtol(1e-9)
                                 .with_matrix_free(true)
                                 .with_log_file("petsc_all_options.log")
                                 .build();
        const auto result  = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationLMVM");
        const auto options = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::LMVM)
                                 .with_max_iterations(500)
                                 .build();
        const auto result  = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        SCOPED_TRACE("OptimizationWithoutGradientIsUnsupported");
        const auto options = solver_options_petsc_builder().build();
        const auto result  = solve(rosenbrock_optimization(false), rosenbrock_start(), *options);

        EXPECT_EQ(result.status, solver_status::unsupported_capability);
    }
    {
        // Exercises the POUNDERS branch of tao_type_string.
        SCOPED_TRACE("LeastSquaresPOUNDERS");
        const auto options = solver_options_petsc_builder()
                                  .with_tao_type(tao_algorithm_enum::POUNDERS)
                                  .with_max_iterations(400)
                                  .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // NLS ("Newton's method with line search") is a general unconstrained
        // minimizer, not a least-squares algorithm: it requires an explicit
        // Hessian (TaoSetHessian) and errors out ("Must call TaoSetHessian()")
        // if driven through the residual/Jacobian routines used for
        // is_least_squares_. Exercise it through the optimization path with
        // an exact Hessian instead, which also covers the NLS branch of
        // tao_type_string.
        SCOPED_TRACE("OptimizationNLSWithExactHessian");
        const auto problem = rosenbrock_optimization_with_hessian();
        const auto options = solver_options_petsc_builder()
                                  .with_tao_type(tao_algorithm_enum::NLS)
                                  .with_max_iterations(400)
                                  .build();
        const auto result = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // NTR with an exact Hessian and matrix_free() false: exercises
        // hessian_tramp and the MatCreateSeqDense/TaoSetHessian branch, which
        // the gradient-only optimization tests never reach.
        SCOPED_TRACE("OptimizationNTRWithExactHessian");
        const auto problem  = rosenbrock_optimization_with_hessian();
        const auto options  = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::NTR)
                                 .with_max_iterations(400)
                                 .build();
        const auto result   = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // Exercises the BQNLS branch of tao_type_string.
        SCOPED_TRACE("OptimizationBQNLS");
        const auto options = solver_options_petsc_builder()
                                  .with_tao_type(tao_algorithm_enum::BQNLS)
                                  .with_max_iterations(400)
                                  .build();
        const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // BNLS ("Bounded Newton Line Search") is a Newton-type bound
        // constrained minimizer: like NLS it needs an explicit Hessian, and
        // (being a bounded variant) is exercised together with explicit
        // bounds that do not exclude the true minimum.
        SCOPED_TRACE("OptimizationBNLSWithExactHessianAndBounds");
        auto problem         = rosenbrock_optimization_with_hessian();
        problem.bounds.lower = {-10.0, -10.0};
        problem.bounds.upper = {10.0, 10.0};
        const auto options   = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::BNLS)
                                 .with_max_iterations(400)
                                 .build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // Both lower and upper bounds set: exercises TaoSetVariableBounds and
        // the non-empty branches of the lower_/upper_ fill loop.
        SCOPED_TRACE("LeastSquaresWithBothBounds");
        auto problem         = rosenbrock_least_squares(true);
        problem.bounds.lower = {-10.0, -10.0};
        problem.bounds.upper = {10.0, 10.0};
        const auto options   = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::BRGN)
                                 .with_max_iterations(400)
                                 .build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // Only a lower bound set (upper empty): exercises the "upper_bounds_
        // empty -> PETSC_INFINITY" ternary branch alongside the non-empty
        // lower branch.
        SCOPED_TRACE("OptimizationWithLowerBoundOnly");
        auto problem         = rosenbrock_optimization(true);
        problem.bounds.lower = {-10.0, -10.0};
        const auto options   = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::LMVM)
                                 .with_max_iterations(400)
                                 .build();
        const auto result    = solve(problem, rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        expect_rosenbrock_minimum(result);
    }
    {
        // A one-iteration budget forces TAO_DIVERGED_MAXITS, exercising the
        // budget_exhausted branch of the TaoConvergedReason classification.
        SCOPED_TRACE("StopsAtMaxIterationsIsBudgetExhausted");
        const auto options = solver_options_petsc_builder()
                                  .with_tao_type(tao_algorithm_enum::BRGN)
                                  .with_max_iterations(1)
                                  .build();
        const auto result = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

        EXPECT_EQ(result.backend, backend::petsc_tao);
        EXPECT_EQ(result.status, solver_status::max_iterations);
    }
}
