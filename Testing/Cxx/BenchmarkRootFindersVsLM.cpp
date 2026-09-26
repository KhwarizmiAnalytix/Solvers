// Benchmark comparing direct root-finding algorithms against
// Levenberg-Marquardt via the problem-structure API.
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
#include "solvers/api/solve.h"
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

template <typename Attempt>
timing_result time_call(Attempt&& attempt, int warmup = 3, int runs = 9)
{
    for (int i = 0; i < warmup; ++i)
        attempt();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i)
    {
        const auto start = clock_type::now();
        attempt();
        const auto end = clock_type::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2]};
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

    // Newton-Raphson.
    {
        double root      = 0.0;
        bool   converged = false;
        const auto timing = time_call([&]() {
            converged = root_finding_algorithms::newton_raphson(
                problem.residual_with_derivative, problem.x2, root, rf_options);
        });
        print_row(problem.name, "RootFinder: Newton", converged,
            std::fabs(root - problem.expected_root), timing);
    }

    // Brent.
    {
        double root      = 0.0;
        bool   converged = false;
        const auto timing = time_call([&]() {
            converged = root_finding_algorithms::brent(
                problem.residual, problem.x1, problem.x2, root, rf_options);
        });
        print_row(problem.name, "RootFinder: Brent", converged,
            std::fabs(root - problem.expected_root), timing);
    }

    // LM via the problem-structure API.
    {
        api::least_squares_problem ls;
        ls.num_parameters = 1;
        ls.num_residuals  = 1;
        ls.residuals = [&](const vector_type& x, vector_type& r) {
            r(0) = problem.residual(x(0));
        };
        ls.jacobian = [&](const vector_type& x, matrix_type& J) {
            double df;
            problem.residual_with_derivative(x(0), df);
            J(0, 0) = df;
        };

        api::solve_options opts;
        opts.algorithm           = api::algorithm::levenberg_marquardt;
        opts.backend             = api::backend::native;
        opts.max_iterations      = 500;
        opts.function_tolerance  = 1e-14;
        opts.parameter_tolerance = 1e-14;

        vector_type x0(1);
        x0 << problem.x2;

        bool converged = false;
        const auto timing = time_call([&]() {
            auto result = api::solve(ls, x0, opts);
            converged   = result.converged();
        });

        auto   result    = api::solve(ls, x0, opts);
        double final_root = result.parameters[0];
        print_row(problem.name, "Solver: LM (API)", result.converged(),
            std::fabs(final_root - problem.expected_root), timing);
    }

    std::cout << "\n";
}

}  // namespace
}  // namespace solverslib

int main()
{
    using namespace solverslib;

    std::cout << "\n========== Root Finders vs Levenberg-Marquardt Benchmark ==========\n";
    std::cout << "Specialized root finders vs LM via the problem-structure API.\n\n";

    print_header();

    for (const auto& problem : testing::make_all_root_finding_test_problems())
        benchmark_problem(problem);

    return 0;
}
