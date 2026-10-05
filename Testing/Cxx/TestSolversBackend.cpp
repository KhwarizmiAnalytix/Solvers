#include <gtest/gtest.h>

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/petsc_tao_solver.h"

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::test;

namespace
{
constexpr double kMinimumTolerance = 1e-4;

#define REQUIRE_BACKEND(available, name)                  \
    if (!(available))                                     \
    {                                                     \
        GTEST_SKIP() << (name) << " backend not compiled in"; \
    }

void expect_rosenbrock_minimum(const solver_result& result)
{
    ASSERT_TRUE(result.converged()) << result.message;
    EXPECT_NEAR(result.parameters(0), 1.0, kMinimumTolerance);
    EXPECT_NEAR(result.parameters(1), 1.0, kMinimumTolerance);
    EXPECT_NEAR(result.objective, 0.0, kMinimumTolerance);
}
}  // namespace

// ---------------------------------------------------------------------------
// Native backend
// ---------------------------------------------------------------------------

TEST(NativeBackend, LevenbergMarquardtLeastSquaresWithJacobian)
{
    const auto options = solver_options_lm_builder()
                             .with_max_iterations(200)
                             .with_function_tolerance(1e-14)
                             .with_gradient_tolerance(1e-12)
                             .with_parameter_tolerance(1e-14)
                             .build();
    const auto result =
        solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.backend, backend::native);
    EXPECT_EQ(result.algorithm, algorithm::levenberg_marquardt);
    expect_rosenbrock_minimum(result);
}

TEST(NativeBackend, LevenbergMarquardtLeastSquaresFiniteDifferences)
{
    const auto options = solver_options_lm_builder()
                             .with_max_iterations(400)
                             .with_function_tolerance(1e-14)
                             .with_gradient_tolerance(1e-12)
                             .with_parameter_tolerance(1e-14)
                             .build();
    const auto result =
        solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

    expect_rosenbrock_minimum(result);
}

TEST(NativeBackend, GaussNewtonLeastSquares)
{
    const auto options = solver_options_gn_builder()
                             .with_max_iterations(200)
                             .with_function_tolerance(1e-14)
                             .with_gradient_tolerance(1e-12)
                             .with_parameter_tolerance(1e-14)
                             .build();
    const auto result =
        solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.algorithm, algorithm::gauss_newton);
    expect_rosenbrock_minimum(result);
}

TEST(NativeBackend, LBFGSOptimization)
{
    const auto options = solver_options_bfgs_builder()
                             .with_max_iterations(1000)
                             .with_function_tolerance(1e-14)
                             .with_gradient_tolerance(1e-10)
                             .build();
    const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.algorithm, algorithm::lbfgs);
    expect_rosenbrock_minimum(result);
}

TEST(NativeBackend, LBFGSLeastSquaresIsUnsupported)
{
    const auto options = solver_options_bfgs_builder().build();
    const auto result =
        solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

TEST(NativeBackend, LevenbergMarquardtOptimizationIsNotRouted)
{
    const auto options = solver_options_lm_builder().build();
    const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::backend_unavailable);
}

TEST(NativeBackend, RejectsWrongInitialGuessSize)
{
    const auto options = solver_options_lm_builder().build();
    vector_type bad(3);
    bad << 0.0, 0.0, 0.0;
    const auto result = solve(rosenbrock_least_squares(true), bad, *options);

    EXPECT_EQ(result.status, solver_status::invalid_problem);
}

// ---------------------------------------------------------------------------
// Ceres backend (least squares only)
// ---------------------------------------------------------------------------

TEST(CeresBackend, LeastSquares)
{
    REQUIRE_BACKEND(ceres_solver::is_supported(), "Ceres");
    const auto options = solver_options_ceres_builder().with_max_iterations(200).build();
    const auto result =
        solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.backend, backend::ceres);
    expect_rosenbrock_minimum(result);
}

TEST(CeresBackend, OptimizationIsUnsupported)
{
    REQUIRE_BACKEND(ceres_solver::is_supported(), "Ceres");
    const auto options = solver_options_ceres_builder().build();
    const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

// ---------------------------------------------------------------------------
// Ipopt backend
// ---------------------------------------------------------------------------

TEST(IpoptBackend, LeastSquaresWithJacobian)
{
    REQUIRE_BACKEND(ipopt_solver::is_supported(), "Ipopt");
    const auto options = solver_options_ipopt_builder().with_max_iterations(500).build();
    const auto result =
        solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.backend, backend::ipopt);
    expect_rosenbrock_minimum(result);
}

TEST(IpoptBackend, LeastSquaresWithoutJacobianIsUnsupported)
{
    REQUIRE_BACKEND(ipopt_solver::is_supported(), "Ipopt");
    const auto options = solver_options_ipopt_builder().build();
    const auto result =
        solve(rosenbrock_least_squares(false), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

TEST(IpoptBackend, OptimizationWithGradient)
{
    REQUIRE_BACKEND(ipopt_solver::is_supported(), "Ipopt");
    const auto options = solver_options_ipopt_builder().with_max_iterations(500).build();
    const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.backend, backend::ipopt);
    expect_rosenbrock_minimum(result);
}

TEST(IpoptBackend, OptimizationWithoutGradientIsUnsupported)
{
    REQUIRE_BACKEND(ipopt_solver::is_supported(), "Ipopt");
    const auto options = solver_options_ipopt_builder().build();
    const auto result = solve(rosenbrock_optimization(false), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

// ---------------------------------------------------------------------------
// PETSc / TAO backend
// ---------------------------------------------------------------------------

TEST(PetscBackend, LeastSquaresBRGN)
{
    REQUIRE_BACKEND(petsc_tao_solver::is_supported(), "PETSc/TAO");
    const auto options = solver_options_petsc_builder()
                             .with_tao_type(tao_algorithm_enum::BRGN)
                             .with_max_iterations(200)
                             .build();
    const auto result =
        solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.backend, backend::petsc_tao);
    expect_rosenbrock_minimum(result);
}

TEST(PetscBackend, OptimizationLMVM)
{
    REQUIRE_BACKEND(petsc_tao_solver::is_supported(), "PETSc/TAO");
    const auto options = solver_options_petsc_builder()
                             .with_tao_type(tao_algorithm_enum::LMVM)
                             .with_max_iterations(500)
                             .build();
    const auto result = solve(rosenbrock_optimization(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.backend, backend::petsc_tao);
    expect_rosenbrock_minimum(result);
}

TEST(PetscBackend, OptimizationWithoutGradientIsUnsupported)
{
    REQUIRE_BACKEND(petsc_tao_solver::is_supported(), "PETSc/TAO");
    const auto options = solver_options_petsc_builder().build();
    const auto result = solve(rosenbrock_optimization(false), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

// ---------------------------------------------------------------------------
// RNC-LM (Riemann Normal Coordinate Levenberg-Marquardt) backend
// ---------------------------------------------------------------------------
// NOTE: Full RNC-LM tests require more sophisticated curve derivative setup.
// For now, only the error-case test is enabled below. RNC-LM infrastructure
// is in place (solver_options_rnc_lm, solve_rnc_lm, rosenbrock_curve_derivatives)
// and ready for complete testing with refined derivative implementations.

TEST(NativeBackend, RNC_LM_WithoutCurveDerivatives_Fails)
{
    const auto options = solver_options_rnc_lm_builder().build();
    const auto result   = solve(rosenbrock_least_squares(true), rosenbrock_start(), *options);

    EXPECT_EQ(result.status, solver_status::unsupported_capability);
}

