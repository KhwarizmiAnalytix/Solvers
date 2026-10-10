// Ceres, Ipopt and PETSc/TAO on the Rosenbrock problem and the raw-SVI
// calibration problem from Ferhati (2020).
//
// Run with: ./BenchmarkExternal [--benchmark_filter=...]

#include <benchmark/benchmark.h>

#include "svi_fixture.h"

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_petsc.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/petsc_tao_solver.h"

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::svi;
using namespace solverslib::test;

namespace
{

const auto ceres_options = solver_options_ceres_builder().with_max_iterations(200).build();

const auto ipopt_options = solver_options_ipopt_builder().with_max_iterations(500).build();

const auto petsc_brgn = solver_options_petsc_builder()
                            .with_tao_type(tao_algorithm_enum::BRGN)
                            .with_max_iterations(200)
                            .build();

const auto petsc_lmvm = solver_options_petsc_builder()
                            .with_tao_type(tao_algorithm_enum::LMVM)
                            .with_max_iterations(500)
                            .build();

const auto svi_ceres_options = solver_options_ceres_builder().with_max_iterations(500).build();

const auto svi_ipopt_options = solver_options_ipopt_builder().with_max_iterations(1000).build();

const auto svi_petsc_brgn = solver_options_petsc_builder()
                                .with_tao_type(tao_algorithm_enum::BRGN)
                                .with_max_iterations(500)
                                .build();

const auto svi_petsc_lmvm = solver_options_petsc_builder()
                                .with_tao_type(tao_algorithm_enum::LMVM)
                                .with_max_iterations(1000)
                                .build();

const auto start = rosenbrock_start();
const auto ls    = rosenbrock_least_squares(true);
const auto opt   = rosenbrock_optimization(true);

}  // namespace

// ============================================================================
// Rosenbrock benchmarks
// ============================================================================

// ---------------------------------------------------------------------------
// Ceres Least Squares
// ---------------------------------------------------------------------------

static void BM_Ceres_LeastSquares(benchmark::State& state)
{
    if (!ceres_solver::is_supported())
    {
        state.SkipWithMessage("Ceres backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *ceres_options));
}
BENCHMARK(BM_Ceres_LeastSquares);

// ---------------------------------------------------------------------------
// Ipopt Least Squares
// ---------------------------------------------------------------------------

static void BM_Ipopt_LeastSquares(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *ipopt_options));
}
BENCHMARK(BM_Ipopt_LeastSquares)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// PETSc/TAO Least Squares
// ---------------------------------------------------------------------------

static void BM_PETSc_BRGN_LeastSquares(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(ls, start, *petsc_brgn));
}
BENCHMARK(BM_PETSc_BRGN_LeastSquares);

// ---------------------------------------------------------------------------
// Ipopt Optimization
// ---------------------------------------------------------------------------

static void BM_Ipopt_Optimization(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(opt, start, *ipopt_options));
}
BENCHMARK(BM_Ipopt_Optimization)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// PETSc/TAO Optimization
// ---------------------------------------------------------------------------

static void BM_PETSc_LMVM_Optimization(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(opt, start, *petsc_lmvm));
}
BENCHMARK(BM_PETSc_LMVM_Optimization);

// ============================================================================
// SVI calibration benchmarks
// ============================================================================

// ---------------------------------------------------------------------------
// Ceres
// ---------------------------------------------------------------------------

static void BM_SVI_Ceres_Analytic(benchmark::State& state)
{
    if (!ceres_solver::is_supported())
    {
        state.SkipWithMessage("Ceres backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_ceres_options));
}
BENCHMARK(BM_SVI_Ceres_Analytic);

static void BM_SVI_Ceres_FiniteDiff(benchmark::State& state)
{
    if (!ceres_solver::is_supported())
    {
        state.SkipWithMessage("Ceres backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_fd, x0, *svi_ceres_options));
}
BENCHMARK(BM_SVI_Ceres_FiniteDiff);

// ---------------------------------------------------------------------------
// Ipopt (optimization form)
// ---------------------------------------------------------------------------

static void BM_SVI_Ipopt_Grad(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_ipopt_options));
}
BENCHMARK(BM_SVI_Ipopt_Grad)->Unit(benchmark::kMillisecond);

static void BM_SVI_Ipopt_FiniteDiff(benchmark::State& state)
{
    if (!ipopt_solver::is_supported())
    {
        state.SkipWithMessage("Ipopt backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_fd, x0, *svi_ipopt_options));
}
BENCHMARK(BM_SVI_Ipopt_FiniteDiff)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// PETSc/TAO BRGN (least squares) and LMVM (optimization)
// ---------------------------------------------------------------------------

static void BM_SVI_PETSc_BRGN_Analytic(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_ls_analytic, x0, *svi_petsc_brgn));
}
BENCHMARK(BM_SVI_PETSc_BRGN_Analytic);

static void BM_SVI_PETSc_LMVM_Grad(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_grad, x0, *svi_petsc_lmvm));
}
BENCHMARK(BM_SVI_PETSc_LMVM_Grad);

static void BM_SVI_PETSc_LMVM_FiniteDiff(benchmark::State& state)
{
    if (!petsc_tao_solver::is_supported())
    {
        state.SkipWithMessage("PETSc/TAO backend not compiled in");
        return;
    }
    const auto& x0 = svi_problem_data().x0;
    for (auto _ : state)
        benchmark::DoNotOptimize(solve(svi_opt_fd, x0, *svi_petsc_lmvm));
}
BENCHMARK(BM_SVI_PETSc_LMVM_FiniteDiff);

BENCHMARK_MAIN();
