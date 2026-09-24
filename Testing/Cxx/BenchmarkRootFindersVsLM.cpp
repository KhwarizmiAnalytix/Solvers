// Benchmark comparing direct root-finding algorithms against
// Levenberg-Marquardt solver formulated as a least-squares problem.
//
// Motivation: finding a root of f(x) = 0 can be approached two ways:
// 1. Direct: use root_finding_algorithms (bisection, newton_raphson, brent, etc.)
// 2. Indirect: minimize ||f(x)||^2 using an optimization solver (LM)
//
// This benchmark demonstrates the trade-off between specialized root finders
// (faster, simpler) and general-purpose optimization (robust, handles residual
// vectors, applicable to least-squares problems).
//
// Build with -DSOLVERS_ENABLE_BENCHMARKS=ON.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "root_finding_test_problems.h"
#include "solver_options/root_finding_options.h"
#include "solver_options/solver_options_lm.h"
#include "solver_wrapper.h"
#include "solvers/levenberg_marquardt_solver.h"
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

// Time a root-finding call (no state to reset, independent repeats).
template <typename Attempt>
timing_result time_root_finder(Attempt&& attempt, int warmup_runs = 3, int timed_runs = 9)
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

// Time a solver call (must reset parameters to initial guess between repeats).
template <typename Attempt>
timing_result time_solver(const root_finding_test_problem& problem,
    std::vector<double>&                                  parameters,
    Attempt&&                                             attempt,
    int                                                   warmup_runs = 2,
    int                                                   timed_runs  = 7)
{
    for (int i = 0; i < warmup_runs; ++i)
    {
        parameters = {problem.x2};  // Single-element vector for 1D problem
        attempt(parameters);
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(timed_runs));
    for (int i = 0; i < timed_runs; ++i)
    {
        parameters       = {problem.x2};
        const auto start = clock_type::now();
        attempt(parameters);
        const auto end = clock_type::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    const double best_ms   = samples_ms.front();
    const double median_ms = samples_ms[samples_ms.size() / 2];
    return {best_ms, median_ms};
}

void print_header()
{
    std::cout << std::left << std::setw(20) << "Problem" << std::setw(20) << "Method"
              << std::setw(12) << "Status" << std::setw(14) << "|error|" << "Timing\n";
    std::cout << std::string(90, '-') << "\n";
}

void print_row(const std::string& problem_name,
    const std::string&            method_name,
    bool                          converged,
    double                        error,
    const timing_result&          timing)
{
    std::cout << std::left << std::setw(20) << problem_name << std::setw(20) << method_name
              << std::setw(12) << (converged ? "converged" : "FAILED") << std::scientific
              << std::setprecision(3) << std::setw(14) << error << std::fixed << "best="
              << std::setprecision(4) << timing.best_ms << "ms median=" << timing.median_ms
              << "ms\n";
}

void benchmark_problem(const root_finding_test_problem& problem)
{
    const root_finding_options rf_options = root_finding_options_builder()
                                                .with_tolerance_function(1e-14)
                                                .with_tolerance_parameter(1e-14)
                                                .with_max_iterations(200)
                                                .build();

    // Direct root finders.
    {
        double root      = 0.0;
        bool   converged = false;

        const auto timing = time_root_finder([&]() {
            converged =
                root_finding_algorithms::newton_raphson(problem.residual_with_derivative, problem.x2, root, rf_options);
        });

        print_row(problem.name, "RootFinder: Newton", converged, std::fabs(root - problem.expected_root), timing);
    }

    {
        double root      = 0.0;
        bool   converged = false;

        const auto timing = time_root_finder([&]() {
            converged = root_finding_algorithms::brent(problem.residual, problem.x1, problem.x2, root, rf_options);
        });

        print_row(problem.name, "RootFinder: Brent", converged, std::fabs(root - problem.expected_root), timing);
    }

    // Levenberg-Marquardt formulated as a least-squares problem.
    // We minimize ||f(x)||^2 by formulating as a single residual r(x) = f(x).
    {
        std::vector<double> parameters = {problem.x2};
        bool                converged   = false;

        // Create residual/jacobian functors for the 1D problem.
        auto residual_func = [&](const vector_type& x, vector_type& r) {
            if (x.size() != 1)
                throw std::invalid_argument("Expected 1 parameter");
            if (r.size() != 1)
                throw std::invalid_argument("Expected 1 residual");
            r(0) = problem.residual(x(0));
        };

        auto jacobian_func = [&](const vector_type& x, matrix_type& J) {
            if (x.size() != 1)
                throw std::invalid_argument("Expected 1 parameter");
            if (J.rows() != 1 || J.cols() != 1)
                throw std::invalid_argument("Expected 1x1 jacobian");
            double df_dx;
            problem.residual_with_derivative(x(0), df_dx);
            J(0, 0) = df_dx;
        };

        solver_wrapper wrapper(1, 1, residual_func, jacobian_func);
        auto            options = std::make_shared<solver_options_lm>(500, 1e-14, 1e-14, 1e-14);

        const auto timing =
            time_solver(problem, parameters, [&](std::vector<double>& p) { converged = wrapper.solve(p, options); });

        double final_root = parameters.empty() ? 0.0 : parameters[0];
        print_row(problem.name, "Solver: LM", converged, std::fabs(final_root - problem.expected_root), timing);
    }

    std::cout << "\n";
}

}  // namespace
}  // namespace solverslib

int main()
{
    using namespace solverslib;

    std::cout << "\n========== Root Finders vs Levenberg-Marquardt Benchmark ==========\n";
    std::cout << "This benchmark compares specialized root finding methods against\n";
    std::cout << "using LM solver to minimize ||f(x)||^2 on scalar root problems.\n\n";

    print_header();

    for (const auto& problem : testing::make_all_root_finding_test_problems())
    {
        benchmark_problem(problem);
    }

    std::cout << "\nKey observations:\n";
    std::cout << "- Root finders (Newton, Brent) are typically faster for scalar problems\n";
    std::cout << "- LM solver adds overhead (Jacobian computation, matrix operations)\n";
    std::cout << "- LM's strength emerges on overdetermined systems (more residuals than parameters)\n";
    std::cout << "- For production code, LM is more robust and handles diverse problem structures\n";

    return 0;
}
