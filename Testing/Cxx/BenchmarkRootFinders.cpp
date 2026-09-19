// Cross-algorithm benchmark: times every root_finding_algorithms method
// (bisection, false_position, ridders, dekker, brent -- bracketing;
// newton_raphson, secant -- open) on a shared set of scalar problems
// (Testing/Cxx/root_finding_test_problems.h).
//
// Timing methodology matches BenchmarkSolvers.cpp: a few discarded warmup
// runs, then several timed repeats reported as best-of and median
// wall-clock time rather than a single noisy sample.
//
// Build with -DSOLVERS_ENABLE_BENCHMARKS=ON.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "root_finding_test_problems.h"
#include "solver_options/root_finding_options.h"
#include "solvers/root_finding_algorithms.h"

namespace solverslib
{
namespace
{
using testing::root_finding_test_problem;
using clock_type = std::chrono::steady_clock;

struct timing_result
{
    double best_ms;
    double median_ms;
};

// Root-finding calls take no in/out state beyond their return values (unlike
// solver_wrapper::solve(), which mutates its parameter vector in place), so
// each repeat is independent - no need to reset anything between runs.
template <typename Attempt>
timing_result time_solve(Attempt&& attempt, int warmup_runs = 3, int timed_runs = 9)
{
    for (int i = 0; i < warmup_runs; ++i)
    {
        attempt();
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(timed_runs));
    for (int i = 0; i < timed_runs; ++i)
    {
        const auto start = clock_type::now();
        attempt();
        const auto end = clock_type::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    const double best_ms   = samples_ms.front();
    const double median_ms = samples_ms[samples_ms.size() / 2];
    return {best_ms, median_ms};
}

void print_row(const std::string& problem_name,
    const std::string&            method_name,
    bool                          converged,
    double                        error,
    const timing_result&          timing)
{
    std::cout << std::left << std::setw(16) << problem_name << std::setw(16) << method_name
              << std::setw(11) << (converged ? "converged" : "FAILED") << std::scientific
              << std::setprecision(3) << std::setw(13) << error << std::fixed << "best="
              << std::setprecision(4) << timing.best_ms << "ms median=" << timing.median_ms
              << "ms\n";
}

template <typename Algorithm>
void run_method(const root_finding_test_problem& problem, const char* name, Algorithm&& algorithm)
{
    double root      = 0.0;
    bool   converged = false;

    const auto timing = time_solve([&]() { converged = algorithm(root); });

    print_row(problem.name, name, converged, std::fabs(root - problem.expected_root), timing);
}

void benchmark_problem(const root_finding_test_problem& problem)
{
    const root_finding_options options = root_finding_options_builder()
                                              .with_tolerance_function(1e-14)
                                              .with_tolerance_parameter(1e-14)
                                              .with_max_iterations(200)
                                              .build();

    // Bracketing, derivative-free.
    run_method(problem, "Bisection", [&](double& root) {
        return root_finding_algorithms::bisection(problem.residual, problem.x1, problem.x2, root, options);
    });
    run_method(problem, "FalsePosition", [&](double& root) {
        return root_finding_algorithms::false_position(
            problem.residual, problem.x1, problem.x2, root, options);
    });
    run_method(problem, "Ridders", [&](double& root) {
        return root_finding_algorithms::ridders(problem.residual, problem.x1, problem.x2, root, options);
    });
    run_method(problem, "Brent", [&](double& root) {
        return root_finding_algorithms::brent(problem.residual, problem.x1, problem.x2, root, options);
    });

    // Bracketing, derivative-based.
    run_method(problem, "Dekker", [&](double& root) {
        return root_finding_algorithms::dekker(
            problem.residual_with_derivative, problem.x1, problem.x2, root, options);
    });

    // Open methods (no bracket requirement, not guaranteed to converge).
    run_method(problem, "NewtonRaphson", [&](double& root) {
        return root_finding_algorithms::newton_raphson(
            problem.residual_with_derivative, problem.x2, root, options);
    });
    run_method(problem, "Secant", [&](double& root) {
        return root_finding_algorithms::secant(problem.residual, problem.x1, problem.x2, root, options);
    });
}

}  // namespace
}  // namespace solverslib

int main()
{
    using namespace solverslib;

    std::cout << std::left << std::setw(16) << "Problem" << std::setw(16) << "Method"
              << std::setw(11) << "Status" << std::setw(13) << "|error|" << "Timing\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& problem : testing::make_all_root_finding_test_problems())
    {
        benchmark_problem(problem);
    }

    return 0;
}
