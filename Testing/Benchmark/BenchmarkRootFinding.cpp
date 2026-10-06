// Benchmark every root-finding algorithm and polynomial solver.
//
// Run with: ./BenchmarkRootFinding [--benchmark_filter=...] [--benchmark_min_time=...]

#include <benchmark/benchmark.h>

#include <cmath>

#include "solver_options/root_finding_options.h"
#include "solvers/polynomial_solver.h"
#include "solvers/root_finding_algorithms.h"

using namespace solverslib;
using namespace solverslib::detail;

namespace
{
const auto opts = root_finding_options_builder()
                      .with_tolerance_function(1e-12)
                      .with_tolerance_parameter(1e-12)
                      .with_max_iterations(100)
                      .build();

// f(x) = x² - 2,  root at √2 ∈ [1, 2]
double f_val(double x)
{
    return x * x - 2.0;
}

double f_grad(double x, double& df)
{
    df = 2.0 * x;
    return x * x - 2.0;
}
}  // namespace

// ---------------------------------------------------------------------------
// Bracketing methods
// ---------------------------------------------------------------------------

static void BM_Bisection(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_bisection(f_val, 1.0, 2.0, opts));
}
BENCHMARK(BM_Bisection);

static void BM_FalsePosition(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_false_position(f_val, 1.0, 2.0, opts));
}
BENCHMARK(BM_FalsePosition);

static void BM_Ridders(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_ridders(f_val, 1.0, 2.0, opts));
}
BENCHMARK(BM_Ridders);

static void BM_Brent(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_brent(f_val, 1.0, 2.0, opts));
}
BENCHMARK(BM_Brent);

static void BM_Dekker(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_dekker(f_grad, 1.0, 2.0, opts));
}
BENCHMARK(BM_Dekker);

// ---------------------------------------------------------------------------
// Open methods
// ---------------------------------------------------------------------------

static void BM_NewtonRaphson(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_newton_raphson(f_grad, 1.5, opts));
}
BENCHMARK(BM_NewtonRaphson);

static void BM_Secant(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_secant(f_val, 1.0, 2.0, opts));
}
BENCHMARK(BM_Secant);

// ---------------------------------------------------------------------------
// Polynomial solvers
// ---------------------------------------------------------------------------

static void BM_SecondDegreePolynomial(benchmark::State& state)
{
    // x² - 3x + 2 = 0 → min positive root = 1
    for (auto _ : state)
        benchmark::DoNotOptimize(polynomial_solver::second_degree_polynomial_solver(-3.0, 2.0));
}
BENCHMARK(BM_SecondDegreePolynomial);

static void BM_ThirdDegreePolynomial(benchmark::State& state)
{
    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6 → largest root = 3
    for (auto _ : state)
        benchmark::DoNotOptimize(
            polynomial_solver::third_degree_polynomial_solver(-6.0, 11.0, -6.0));
}
BENCHMARK(BM_ThirdDegreePolynomial);

static void BM_FourthDegreePolynomial(benchmark::State& state)
{
    // (x-1)(x-2)(x-3)(x-4) = x⁴ - 10x³ + 35x² - 50x + 24 → largest root = 4
    for (auto _ : state)
        benchmark::DoNotOptimize(
            polynomial_solver::fourth_degree_polynomial_solver(-10.0, 35.0, -50.0, 24.0));
}
BENCHMARK(BM_FourthDegreePolynomial);

BENCHMARK_MAIN();
