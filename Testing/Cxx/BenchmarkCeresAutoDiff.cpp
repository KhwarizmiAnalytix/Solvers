#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

#if SOLVERS_HAS_CERES

#include "optimization_test_problems.h"
#include "solvers/integrations/autodiff_provider.h"
#include "solvers/api/solve.h"

namespace solverslib
{
namespace
{

// Helper to measure time and iterations
struct BenchmarkResult
{
    std::string name;
    double wall_time_ms = 0.0;
    int iterations = 0;
    size_t residual_evals = 0;
    size_t jacobian_evals = 0;
    double final_objective = 0.0;
    bool converged = false;
    std::string derivative_source;

    void print_header()
    {
        std::cout << std::setw(30) << "Derivative Mode"
                  << " | " << std::setw(12) << "Time (ms)"
                  << " | " << std::setw(10) << "Iters"
                  << " | " << std::setw(12) << "Objective"
                  << " | " << std::setw(8) << "Conv"
                  << " | " << std::setw(20) << "Actual Source"
                  << std::endl;
        std::cout << std::string(115, '-') << std::endl;
    }

    void print_row() const
    {
        std::cout << std::setw(30) << name
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << wall_time_ms
                  << " | " << std::setw(10) << iterations
                  << " | " << std::setw(12) << std::scientific << std::setprecision(3) << final_objective
                  << " | " << std::setw(8) << (converged ? "YES" : "NO")
                  << " | " << std::setw(20) << derivative_source
                  << std::endl;
    }
};

void benchmark_rosenbrock()
{
    std::cout << "\n========== ROSENBROCK (2D, 2 residuals) ==========" << std::endl;

    const auto& tp = testing::make_rosenbrock_problem();

    // Create analytical Jacobian problem
    api::least_squares_problem analytical_problem;
    analytical_problem.num_parameters = tp.num_parameters;
    analytical_problem.num_residuals = tp.num_residuals;
    analytical_problem.residuals = tp.residuals;
    analytical_problem.jacobian = tp.jacobian;

    // Create AD problem
    auto ad_problem = least_squares(testing::RosenbrocResiduals{}, tp.num_parameters, tp.num_residuals);
    ad_problem.derivatives(auto_diff());

    vector_type x0 = to_vector_type(tp.initial_guess);
    api::solve_options opts;
    opts.backend = api::backend::ceres;
    opts.max_iterations = 100;

    BenchmarkResult::BenchmarkResult().print_header();

    // Analytical Jacobian
    {
        auto opts_analytical = opts;
        opts_analytical.derivatives = api::derivative_mode::supplied;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(analytical_problem, x0, opts_analytical);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Analytical Jacobian";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }

    // Automatic Differentiation
    {
        auto opts_ad = opts;
        opts_ad.derivatives = api::derivative_mode::automatic_differentiation;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(ad_problem, x0, opts_ad);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Ceres AD (stride=4)";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }

    // Numerical Differentiation (finite difference)
    {
        auto opts_fd = opts;
        opts_fd.derivatives = api::derivative_mode::finite_difference;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(ad_problem, x0, opts_fd);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Finite Differences";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }
}

void benchmark_powell_singular()
{
    std::cout << "\n========== POWELL SINGULAR (4D, 4 residuals) ==========" << std::endl;

    const auto& tp = testing::make_powell_singular_problem();

    // Create analytical Jacobian problem
    api::least_squares_problem analytical_problem;
    analytical_problem.num_parameters = tp.num_parameters;
    analytical_problem.num_residuals = tp.num_residuals;
    analytical_problem.residuals = tp.residuals;
    analytical_problem.jacobian = tp.jacobian;

    // Create AD problem
    auto ad_problem = least_squares(testing::PowellSingularResiduals{}, tp.num_parameters, tp.num_residuals);
    ad_problem.derivatives(auto_diff());

    vector_type x0 = to_vector_type(tp.initial_guess);
    api::solve_options opts;
    opts.backend = api::backend::ceres;
    opts.max_iterations = 200;

    BenchmarkResult::BenchmarkResult().print_header();

    // Analytical Jacobian
    {
        auto opts_analytical = opts;
        opts_analytical.derivatives = api::derivative_mode::supplied;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(analytical_problem, x0, opts_analytical);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Analytical Jacobian";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }

    // Automatic Differentiation
    {
        auto opts_ad = opts;
        opts_ad.derivatives = api::derivative_mode::automatic_differentiation;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(ad_problem, x0, opts_ad);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Ceres AD (stride=4)";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }

    // Numerical Differentiation
    {
        auto opts_fd = opts;
        opts_fd.derivatives = api::derivative_mode::finite_difference;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(ad_problem, x0, opts_fd);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Finite Differences";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }
}

void benchmark_exponential_fit()
{
    std::cout << "\n========== EXPONENTIAL FIT (2D, 10 residuals) ==========" << std::endl;

    const auto& tp = testing::make_exponential_fit_problem();

    // Create analytical Jacobian problem
    api::least_squares_problem analytical_problem;
    analytical_problem.num_parameters = tp.num_parameters;
    analytical_problem.num_residuals = tp.num_residuals;
    analytical_problem.residuals = tp.residuals;
    analytical_problem.jacobian = tp.jacobian;

    // Create AD problem
    auto ad_problem = least_squares(testing::ExponentialFitResiduals{}, tp.num_parameters, tp.num_residuals);
    ad_problem.derivatives(auto_diff());

    vector_type x0 = to_vector_type(tp.initial_guess);
    api::solve_options opts;
    opts.backend = api::backend::ceres;
    opts.max_iterations = 200;

    BenchmarkResult::BenchmarkResult().print_header();

    // Analytical Jacobian
    {
        auto opts_analytical = opts;
        opts_analytical.derivatives = api::derivative_mode::supplied;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(analytical_problem, x0, opts_analytical);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Analytical Jacobian";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }

    // Automatic Differentiation
    {
        auto opts_ad = opts;
        opts_ad.derivatives = api::derivative_mode::automatic_differentiation;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(ad_problem, x0, opts_ad);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Ceres AD (stride=4)";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }

    // Numerical Differentiation
    {
        auto opts_fd = opts;
        opts_fd.derivatives = api::derivative_mode::finite_difference;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = api::solve(ad_problem, x0, opts_fd);
        auto end = std::chrono::high_resolution_clock::now();

        BenchmarkResult bench;
        bench.name = "Finite Differences";
        bench.wall_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        bench.iterations = result.iterations;
        bench.residual_evals = result.residual_evaluations;
        bench.jacobian_evals = result.jacobian_evaluations;
        bench.final_objective = result.objective;
        bench.converged = result.converged();
        if (result.effective_derivative_source)
        {
            bench.derivative_source = api::to_string(*result.effective_derivative_source);
        }
        bench.print_row();
    }
}

}  // namespace
}  // namespace solverslib

int main()
{
    std::cout << "Ceres Automatic Differentiation Benchmark\n"
              << "Comparing Analytical vs AD vs Numerical Jacobians\n"
              << std::endl;

    solverslib::benchmark_rosenbrock();
    solverslib::benchmark_powell_singular();
    solverslib::benchmark_exponential_fit();

    std::cout << "\nNote: Times include solver overhead; focus on relative comparison.\n"
              << "For production use, measure on your actual models.\n";

    return 0;
}

#else

int main()
{
    std::cerr << "Ceres not enabled (SOLVERS_ENABLE_CERES=OFF)\n";
    return 1;
}

#endif
