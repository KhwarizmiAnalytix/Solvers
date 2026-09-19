// Cross-backend benchmark: times the native Levenberg-Marquardt and L-BFGS
// solvers against the optional Ceres and NLopt backends on a shared set of
// least-squares problems (Testing/Cxx/optimization_test_problems.h).
//
// Timing methodology follows the same idea as Eigen's bench/BenchTimer.h and
// PyTorch's torch.utils.benchmark.Timer: a few discarded warmup runs to prime
// caches/allocations, then several timed repeats reported as best-of and
// median wall-clock time rather than a single noisy sample.
//
// Build with -DSOLVERS_ENABLE_BENCHMARKS=ON (and -DSOLVERS_ENABLE_CERES=ON
// -DSOLVERS_ENABLE_NLOPT=ON to include those backends in the comparison).
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "optimization_test_problems.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_nlopt.h"
#include "solver_wrapper.h"
#include "solvers/ceres_solver.h"
#include "solvers/nlopt_solver.h"

namespace solverslib
{
namespace
{
using testing::optimization_test_problem;
using clock_type = std::chrono::steady_clock;

struct timing_result
{
    double best_ms;
    double median_ms;
};

double residual_norm(
    const optimization_test_problem& problem, const std::vector<double>& parameters)
{
    vector_type x = to_vector_type(parameters);
    vector_type r = make_vector(problem.num_residuals);
    problem.residuals(x, r);
    return r.norm();
}

// Runs `attempt` repeatedly, resetting `parameters` to the problem's initial
// guess before every call since solve() mutates it in place.
template <typename Attempt>
timing_result time_solve(const optimization_test_problem& problem,
    std::vector<double>&                                  parameters,
    Attempt&&                                             attempt,
    int                                                   warmup_runs = 2,
    int                                                   timed_runs  = 7)
{
    for (int i = 0; i < warmup_runs; ++i)
    {
        parameters = problem.initial_guess;
        attempt(parameters);
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(timed_runs));
    for (int i = 0; i < timed_runs; ++i)
    {
        parameters       = problem.initial_guess;
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

void print_row(const std::string& problem_name,
    const std::string&            solver_name,
    bool                          available,
    bool                          converged,
    double                        final_residual,
    const timing_result&          timing)
{
    std::cout << std::left << std::setw(16) << problem_name << std::setw(17) << solver_name;
    if (!available)
    {
        std::cout << std::setw(11) << "n/a" << std::setw(16) << "-" << "not compiled in\n";
        return;
    }

    std::cout << std::setw(11) << (converged ? "converged" : "FAILED") << std::setw(16)
              << final_residual << "best=" << std::fixed << std::setprecision(4) << timing.best_ms
              << "ms median=" << timing.median_ms << "ms\n";
}

void benchmark_problem(const optimization_test_problem& problem)
{
    std::vector<double> parameters = problem.initial_guess;

    // Native Levenberg-Marquardt.
    {
        solver_wrapper wrapper(
            problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
        auto       options   = std::make_shared<solver_options_lm>(500, 1e-14, 1e-14, 1e-14);
        bool       converged = false;
        const auto timing    = time_solve(problem,
            parameters,
            [&](std::vector<double>& p) { converged = wrapper.solve(p, options); });
        print_row(problem.name,
            "LM (native)",
            true,
            converged,
            residual_norm(problem, parameters),
            timing);
    }

    // Native L-BFGS.
    {
        solver_wrapper wrapper(
            problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
        auto       options   = std::make_shared<solver_options_bfgs>(500, 1e-14, 1e-14, 1e-14);
        bool       converged = false;
        const auto timing    = time_solve(problem,
            parameters,
            [&](std::vector<double>& p) { converged = wrapper.solve(p, options); });
        print_row(problem.name,
            "LBFGS (native)",
            true,
            converged,
            residual_norm(problem, parameters),
            timing);
    }

    // NLopt (LD_LBFGS), optional backend.
    {
        const bool    available = nlopt_solver::is_supported();
        bool          converged = false;
        timing_result timing{0.0, 0.0};
        if (available)
        {
            solver_wrapper wrapper(
                problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
            auto options = std::make_shared<solver_options_nlopt>(
                nlopt_algo_name_enum::LBFGS, 500, 1e-14, 1e-14, 1e-14);
            timing = time_solve(problem,
                parameters,
                [&](std::vector<double>& p) { converged = wrapper.solve(p, options); });
        }
        print_row(problem.name,
            "NLopt",
            available,
            converged,
            residual_norm(problem, parameters),
            timing);
    }

    // Ceres, optional backend.
    {
        const bool    available = ceres_solver::is_supported();
        bool          converged = false;
        timing_result timing{0.0, 0.0};
        if (available)
        {
            solver_wrapper wrapper(
                problem.num_parameters, problem.num_residuals, problem.residuals, problem.jacobian);
            auto options = std::make_shared<solver_options_ceres>(500, 1e-14, 1e-14, 1e-14);
            timing       = time_solve(problem,
                parameters,
                [&](std::vector<double>& p) { converged = wrapper.solve(p, options); });
        }
        print_row(problem.name,
            "Ceres",
            available,
            converged,
            residual_norm(problem, parameters),
            timing);
    }
}

}  // namespace
}  // namespace solverslib

int main()
{
    using namespace solverslib;

    std::cout << std::left << std::setw(16) << "Problem" << std::setw(17) << "Solver"
              << std::setw(11) << "Status" << std::setw(16) << "||residual||" << "Timing\n";
    std::cout << std::string(93, '-') << "\n";

    for (const auto& problem : testing::make_all_test_problems())
    {
        benchmark_problem(problem);
    }

    return 0;
}
