// Benchmark every backend reachable through api::solve() on the Rosenbrock problem.
// Uses Google Benchmark for reliable timing and statistics.
//
// Run with: ./BenchmarkSolverComparison [--benchmark_filter=...] [--benchmark_min_time=...]

#include <benchmark/benchmark.h>

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_petsc.h"
#include "solvers/ceres_solver.h"
#include "solvers/ipopt_solver.h"
#include "solvers/petsc_tao_solver.h"

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::test;

namespace
{
const auto lm_options = solver_options_lm_builder()
                            .with_max_iterations(400)
                            .with_function_tolerance(1e-14)
                            .with_gradient_tolerance(1e-12)
                            .with_parameter_tolerance(1e-14)
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

const auto start = rosenbrock_start();
const auto ls    = rosenbrock_least_squares(true);
const auto ls_fd = rosenbrock_least_squares(false);
const auto opt   = rosenbrock_optimization(true);
}  // namespace

// ---------------------------------------------------------------------------
// Native Least Squares
// ---------------------------------------------------------------------------

static void BM_NativeLM_LeastSquares(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(solve(ls, start, *lm_options));
    }
}
BENCHMARK(BM_NativeLM_LeastSquares);

static void BM_NativeLM_FiniteDiff(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(solve(ls_fd, start, *lm_options));
    }
}
BENCHMARK(BM_NativeLM_FiniteDiff);

static void BM_NativeGN_LeastSquares(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(solve(ls, start, *gn_options));
    }
}
BENCHMARK(BM_NativeGN_LeastSquares);

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
    {
        benchmark::DoNotOptimize(solve(ls, start, *ceres_options));
    }
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
    {
        benchmark::DoNotOptimize(solve(ls, start, *ipopt_options));
    }
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
    {
        benchmark::DoNotOptimize(solve(ls, start, *petsc_brgn));
    }
}
BENCHMARK(BM_PETSc_BRGN_LeastSquares);

// ---------------------------------------------------------------------------
// Native Optimization
// ---------------------------------------------------------------------------

static void BM_NativeLBFGS_Optimization(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(solve(opt, start, *bfgs_options));
    }
}
BENCHMARK(BM_NativeLBFGS_Optimization);

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
    {
        benchmark::DoNotOptimize(solve(opt, start, *ipopt_options));
    }
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
    {
        benchmark::DoNotOptimize(solve(opt, start, *petsc_lmvm));
    }
}
BENCHMARK(BM_PETSc_LMVM_Optimization);

BENCHMARK_MAIN();
