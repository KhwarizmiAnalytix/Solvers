// Native solvers on the Rosenbrock problem and the raw-SVI calibration
// problem from Ferhati (2020): Levenberg-Marquardt, Gauss-Newton, L-BFGS
// and RNC-LM.
//
// Run with: ./BenchmarkNative [--benchmark_filter=...]

#include <benchmark/benchmark.h>

#include "svi_fixture.h"

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_rnc_lm.h"
#include "solvers/rnc_lm_solver.h"

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::svi;
using namespace solverslib::test;

namespace
{

// ---------------------------------------------------------------------------
// Rosenbrock options
// ---------------------------------------------------------------------------

const auto lm_options = solver_options_lm_builder()
                            .with_max_iterations(400)
                            .with_function_tolerance(1e-14)
                            .with_gradient_tolerance(1e-12)
                            .with_parameter_tolerance(1e-14)
                            .with_type(levenberg_marquardt_solver_enum::NIELSEN)
                            .build();

const auto lm_options_marquardt = solver_options_lm_builder()
                                       .with_max_iterations(400)
                                       .with_function_tolerance(1e-14)
                                       .with_gradient_tolerance(1e-12)
                                       .with_parameter_tolerance(1e-14)
                                       .with_type(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT)
                                       .build();

const auto lm_options_quadratic = solver_options_lm_builder()
                                       .with_max_iterations(400)
                                       .with_function_tolerance(1e-14)
                                       .with_gradient_tolerance(1e-12)
                                       .with_parameter_tolerance(1e-14)
                                       .with_type(levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION)
                                       .build();

const auto gn_options = solver_options_gn_builder()
                            .with_max_iterations(200)
                            .with_function_tolerance(1e-14)
                            .with_gradient_tolerance(1e-12)
                            .with_parameter_tolerance(1e-14)
                            .build();

const auto bfgs_options = solver_options_bfgs_builder()
                              .with_max_iterations(1000)
                              .with_function_tolerance(1e-14)
                              .with_gradient_tolerance(1e-10)
                              .build();

// ---------------------------------------------------------------------------
// SVI-specific LM option variants
// ---------------------------------------------------------------------------

// Same budget and geodesic/bold pair for Nielsen, Marquardt, and quadratic interpolation.
std::shared_ptr<const solver_options_lm> make_svi_lm(
    levenberg_marquardt_solver_enum type, bool geodesic, bool bold)
{
    return solver_options_lm_builder()
        .with_max_iterations(1000)
        .with_function_tolerance(1e-12)
        .with_gradient_tolerance(1e-10)
        .with_parameter_tolerance(1e-12)
        .with_geodesic_acceleration(geodesic)
        .with_bold_acceptance(bold)
        .with_type(type)
        .build();
}

// Geodesic on, bold acceptance off.
const auto svi_lm_default = make_svi_lm(levenberg_marquardt_solver_enum::NIELSEN, true, false);
const auto svi_lm_marquardt =
    make_svi_lm(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT, true, false);
const auto svi_lm_quadratic =
    make_svi_lm(levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION, true, false);

// Geodesic off, bold acceptance off.
const auto svi_lm_no_geodesic = make_svi_lm(levenberg_marquardt_solver_enum::NIELSEN, false, false);
const auto svi_lm_no_geodesic_marquardt =
    make_svi_lm(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT, false, false);
const auto svi_lm_no_geodesic_quadratic =
    make_svi_lm(levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION, false, false);

// Geodesic on, bold acceptance on.
const auto svi_lm_bold = make_svi_lm(levenberg_marquardt_solver_enum::NIELSEN, true, true);
const auto svi_lm_bold_marquardt =
    make_svi_lm(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT, true, true);
const auto svi_lm_bold_quadratic =
    make_svi_lm(levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION, true, true);

// Geodesic off, bold acceptance on.
const auto svi_lm_bold_no_geodesic =
    make_svi_lm(levenberg_marquardt_solver_enum::NIELSEN, false, true);
const auto svi_lm_bold_no_geodesic_marquardt =
    make_svi_lm(levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT, false, true);
const auto svi_lm_bold_no_geodesic_quadratic =
    make_svi_lm(levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION, false, true);

const auto svi_gn_options = solver_options_gn_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .build();

// ---------------------------------------------------------------------------
// SVI RNC-LM options — vary order (1–4)
// ---------------------------------------------------------------------------

const auto svi_rnc_order1 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(1)
                                .build();

const auto svi_rnc_order2 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(2)
                                .build();

const auto svi_rnc_order3 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(3)
                                .build();

const auto svi_rnc_order4 = solver_options_rnc_lm_builder()
                                .with_max_iterations(1000)
                                .with_function_tolerance(1e-12)
                                .with_gradient_tolerance(1e-10)
                                .with_parameter_tolerance(1e-12)
                                .with_order(4)
                                .build();

// ---------------------------------------------------------------------------
// SVI L-BFGS options — vary line search method/type and memory size (tau)
// ---------------------------------------------------------------------------

// Default: Armijo + Nocedal-Wright, tau=6
const auto svi_lbfgs_armijo = solver_options_bfgs_builder()
                                   .with_max_iterations(1000)
                                   .with_function_tolerance(1e-12)
                                   .with_gradient_tolerance(1e-10)
                                   .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                   .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                   .with_tau(6)
                                   .build();

const auto svi_lbfgs_wolfe = solver_options_bfgs_builder()
                                  .with_max_iterations(1000)
                                  .with_function_tolerance(1e-12)
                                  .with_gradient_tolerance(1e-10)
                                  .with_method_type(lbfgs_line_search_method_type::WOLFE)
                                  .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                  .with_tau(6)
                                  .build();

const auto svi_lbfgs_strong_wolfe = solver_options_bfgs_builder()
                                        .with_max_iterations(1000)
                                        .with_function_tolerance(1e-12)
                                        .with_gradient_tolerance(1e-10)
                                        .with_method_type(lbfgs_line_search_method_type::STRONG_WOLFE)
                                        .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                        .with_tau(6)
                                        .build();

const auto svi_lbfgs_backtracking = solver_options_bfgs_builder()
                                        .with_max_iterations(1000)
                                        .with_function_tolerance(1e-12)
                                        .with_gradient_tolerance(1e-10)
                                        .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                        .with_type(lbfgs_line_search_type::BACKTRACKING)
                                        .with_tau(6)
                                        .build();

const auto svi_lbfgs_bracketing = solver_options_bfgs_builder()
                                       .with_max_iterations(1000)
                                       .with_function_tolerance(1e-12)
                                       .with_gradient_tolerance(1e-10)
                                       .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                       .with_type(lbfgs_line_search_type::BRACKETING)
                                       .with_tau(6)
                                       .build();

const auto svi_lbfgs_tau3 = solver_options_bfgs_builder()
                                 .with_max_iterations(1000)
                                 .with_function_tolerance(1e-12)
                                 .with_gradient_tolerance(1e-10)
                                 .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                 .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                 .with_tau(3)
                                 .build();

const auto svi_lbfgs_tau10 = solver_options_bfgs_builder()
                                  .with_max_iterations(1000)
                                  .with_function_tolerance(1e-12)
                                  .with_gradient_tolerance(1e-10)
                                  .with_method_type(lbfgs_line_search_method_type::ARMIJO)
                                  .with_type(lbfgs_line_search_type::NOCEDAL_WRIGHT)
                                  .with_tau(10)
                                  .build();

const auto svi_rnc_problem = make_svi_rnc_ls();

const auto start = rosenbrock_start();
const auto ls    = rosenbrock_least_squares(true);
const auto ls_fd = rosenbrock_least_squares(false);
const auto opt   = rosenbrock_optimization(true);

}  // namespace

// ============================================================================
// Rosenbrock benchmarks
// ============================================================================

// ---------------------------------------------------------------------------
// Native Least Squares
// ---------------------------------------------------------------------------

static void BM_NativeLM_LeastSquares(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *lm_options));
}
BENCHMARK(BM_NativeLM_LeastSquares);

static void BM_NativeLM_FiniteDiff(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls_fd, start, *lm_options));
}
BENCHMARK(BM_NativeLM_FiniteDiff);

static void BM_NativeLM_LeastSquares_Marquardt(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *lm_options_marquardt));
}
BENCHMARK(BM_NativeLM_LeastSquares_Marquardt);

static void BM_NativeLM_FiniteDiff_Marquardt(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls_fd, start, *lm_options_marquardt));
}
BENCHMARK(BM_NativeLM_FiniteDiff_Marquardt);

static void BM_NativeLM_LeastSquares_Quadratic(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *lm_options_quadratic));
}
BENCHMARK(BM_NativeLM_LeastSquares_Quadratic);

static void BM_NativeLM_FiniteDiff_Quadratic(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls_fd, start, *lm_options_quadratic));
}
BENCHMARK(BM_NativeLM_FiniteDiff_Quadratic);

static void BM_NativeGN_LeastSquares(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *gn_options));
}
BENCHMARK(BM_NativeGN_LeastSquares);

// ---------------------------------------------------------------------------
// Native Optimization
// ---------------------------------------------------------------------------

static void BM_NativeLBFGS_Optimization(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(opt, start, *bfgs_options));
}
BENCHMARK(BM_NativeLBFGS_Optimization);

// ============================================================================
// SVI calibration benchmarks
// ============================================================================

// ---------------------------------------------------------------------------
// Native LM — analytic Jacobian variants
// ---------------------------------------------------------------------------

static void BM_SVI_NativeLM_Analytic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_default));
}
BENCHMARK(BM_SVI_NativeLM_Analytic);

static void BM_SVI_NativeLM_Analytic_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_NoGeodesic);

static void BM_SVI_NativeLM_Analytic_Bold(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_bold));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold);

static void BM_SVI_NativeLM_Analytic_Bold_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_bold_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold_NoGeodesic);

static void BM_SVI_NativeLM_Analytic_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Marquardt);

static void BM_SVI_NativeLM_Analytic_NoGeodesic_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_no_geodesic_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_NoGeodesic_Marquardt);

static void BM_SVI_NativeLM_Analytic_Bold_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_bold_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold_Marquardt);

static void BM_SVI_NativeLM_Analytic_Bold_NoGeodesic_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_bold_no_geodesic_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold_NoGeodesic_Marquardt);

static void BM_SVI_NativeLM_Analytic_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Quadratic);

static void BM_SVI_NativeLM_Analytic_NoGeodesic_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_no_geodesic_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_NoGeodesic_Quadratic);

static void BM_SVI_NativeLM_Analytic_Bold_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_bold_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold_Quadratic);

static void BM_SVI_NativeLM_Analytic_Bold_NoGeodesic_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_analytic, x0, *svi_lm_bold_no_geodesic_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_Analytic_Bold_NoGeodesic_Quadratic);

// ---------------------------------------------------------------------------
// Native LM — finite differences
// ---------------------------------------------------------------------------

static void BM_SVI_NativeLM_FiniteDiff(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_default));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff);

static void BM_SVI_NativeLM_FiniteDiff_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_NoGeodesic);

static void BM_SVI_NativeLM_FiniteDiff_Bold(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_bold));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold);

static void BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_bold_no_geodesic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic);

static void BM_SVI_NativeLM_FiniteDiff_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Marquardt);

static void BM_SVI_NativeLM_FiniteDiff_NoGeodesic_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_no_geodesic_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_NoGeodesic_Marquardt);

static void BM_SVI_NativeLM_FiniteDiff_Bold_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_bold_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold_Marquardt);

static void BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic_Marquardt(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_bold_no_geodesic_marquardt));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic_Marquardt);

static void BM_SVI_NativeLM_FiniteDiff_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Quadratic);

static void BM_SVI_NativeLM_FiniteDiff_NoGeodesic_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_no_geodesic_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_NoGeodesic_Quadratic);

static void BM_SVI_NativeLM_FiniteDiff_Bold_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_bold_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold_Quadratic);

static void BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic_Quadratic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_unbounded_fd, x0, *svi_lm_bold_no_geodesic_quadratic));
}
BENCHMARK(BM_SVI_NativeLM_FiniteDiff_Bold_NoGeodesic_Quadratic);

// ---------------------------------------------------------------------------
// Native GN
// ---------------------------------------------------------------------------

static void BM_SVI_NativeGN_Analytic(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_gn_options));
}
BENCHMARK(BM_SVI_NativeGN_Analytic);

static void BM_SVI_NativeGN_FiniteDiff(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_gn_options));
}
BENCHMARK(BM_SVI_NativeGN_FiniteDiff);

// ============================================================================
// SVI RNC-LM benchmarks — vary curve order
// ============================================================================

static void BM_SVI_RncLM_Order1(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order1));
}
BENCHMARK(BM_SVI_RncLM_Order1);

static void BM_SVI_RncLM_Order2(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order2));
}
BENCHMARK(BM_SVI_RncLM_Order2);

static void BM_SVI_RncLM_Order3(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order3));
}
BENCHMARK(BM_SVI_RncLM_Order3);

static void BM_SVI_RncLM_Order4(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve_rnc_lm(svi_rnc_problem, x0, *svi_rnc_order4));
}
BENCHMARK(BM_SVI_RncLM_Order4);

// ============================================================================
// SVI L-BFGS benchmarks — vary line search method, type, memory size
// ============================================================================

static void BM_SVI_LBFGS_Armijo_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_armijo));
}
BENCHMARK(BM_SVI_LBFGS_Armijo_Grad);

static void BM_SVI_LBFGS_Wolfe_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_wolfe));
}
BENCHMARK(BM_SVI_LBFGS_Wolfe_Grad);

static void BM_SVI_LBFGS_StrongWolfe_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_strong_wolfe));
}
BENCHMARK(BM_SVI_LBFGS_StrongWolfe_Grad);

static void BM_SVI_LBFGS_Backtracking_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_backtracking));
}
BENCHMARK(BM_SVI_LBFGS_Backtracking_Grad);

static void BM_SVI_LBFGS_Bracketing_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_bracketing));
}
BENCHMARK(BM_SVI_LBFGS_Bracketing_Grad);

static void BM_SVI_LBFGS_Armijo_FiniteDiff(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_fd, x0, *svi_lbfgs_armijo));
}
BENCHMARK(BM_SVI_LBFGS_Armijo_FiniteDiff);

static void BM_SVI_LBFGS_Tau3_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_tau3));
}
BENCHMARK(BM_SVI_LBFGS_Tau3_Grad);

static void BM_SVI_LBFGS_Tau10_Grad(benchmark::State& state)
{
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_lbfgs_tau10));
}
BENCHMARK(BM_SVI_LBFGS_Tau10_Grad);

BENCHMARK_MAIN();
