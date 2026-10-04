#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if SOLVERS_HAS_CERES

#include "optimization_test_problems.h"
#include "solvers/api/solve.h"
#include "solvers/integrations/autodiff_provider.h"

namespace solverslib
{
namespace
{

constexpr std::size_t kWarmupRounds = 3;
constexpr std::size_t kSamples      = 21;
constexpr std::size_t kBatchSize    = 50;
constexpr std::size_t kNumCases     = 4;

struct BenchmarkCase
{
    std::string                         name;
    api::derivative_mode                expected_source;
    std::function<api::solver_result()> solve;
};

struct BenchmarkResult
{
    std::string name;
    double      median_time_us  = 0.0;
    int         iterations      = 0;
    double      final_objective = 0.0;
    bool        converged       = false;
    std::string derivative_source;

    static void print_header()
    {
        std::cout << std::setw(30) << "Derivative Mode"
                  << " | " << std::setw(17) << "Median (us/solve)"
                  << " | " << std::setw(10) << "Iters"
                  << " | " << std::setw(12) << "Objective"
                  << " | " << std::setw(8) << "Conv"
                  << " | " << std::setw(26) << "Actual Source" << '\n';
        std::cout << std::string(124, '-') << '\n';
    }

    void print_row() const
    {
        std::cout << std::setw(30) << name << " | " << std::setw(17) << std::fixed
                  << std::setprecision(3) << median_time_us << " | " << std::setw(10) << iterations
                  << " | " << std::setw(12) << std::scientific << std::setprecision(3)
                  << final_objective << " | " << std::setw(8) << (converged ? "YES" : "NO") << " | "
                  << std::setw(26) << derivative_source << '\n';
    }
};

using Cases   = std::array<BenchmarkCase, kNumCases>;
using Results = std::array<BenchmarkResult, kNumCases>;

struct DenseNonlinearResiduals
{
    std::size_t         num_parameters;
    std::size_t         num_residuals;
    std::vector<double> coefficients;
    std::vector<double> targets;

    template <typename T> bool operator()(const T* const x, T* residuals) const
    {
        for (std::size_t i = 0; i < num_residuals; ++i)
        {
            T value = T(0.0);
            for (std::size_t j = 0; j < num_parameters; ++j)
            {
                const T xj = x[j];
                value += T(coefficients[i * num_parameters + j]) * (sin(xj) + T(0.05) * xj * xj);
            }
            residuals[i] = value - T(targets[i]);
        }
        return true;
    }
};

std::pair<testing::optimization_test_problem, DenseNonlinearResiduals>
make_dense_nonlinear_problem()
{
    constexpr std::size_t num_parameters = 16;
    constexpr std::size_t num_residuals  = 128;

    std::vector<double> coefficients(num_parameters * num_residuals);
    std::uint32_t       state = 0x12345678U;
    for (double& coefficient : coefficients)
    {
        state       = state * 1664525U + 1013904223U;
        coefficient = static_cast<double>((state >> 8U) % 2001U) / 1000.0 - 1.0;
    }

    std::vector<double> expected_solution(num_parameters);
    for (std::size_t j = 0; j < num_parameters; ++j)
    {
        expected_solution[j] = 0.02 * static_cast<double>(j + 1);
    }

    std::vector<double> targets(num_residuals, 0.0);
    for (std::size_t i = 0; i < num_residuals; ++i)
    {
        for (std::size_t j = 0; j < num_parameters; ++j)
        {
            const double xj = expected_solution[j];
            targets[i] += coefficients[i * num_parameters + j] * (std::sin(xj) + 0.05 * xj * xj);
        }
    }

    DenseNonlinearResiduals functor{num_parameters, num_residuals, coefficients, targets};

    testing::optimization_test_problem problem;
    problem.name              = "DenseNonlinear";
    problem.num_parameters    = num_parameters;
    problem.num_residuals     = num_residuals;
    problem.initial_guess     = std::vector<double>(num_parameters, 0.0);
    problem.expected_solution = expected_solution;
    problem.residuals         = [functor](const vector_type& x, vector_type& residuals)
    { functor(x.data(), residuals.data()); };
    problem.jacobian = [coefficients](const vector_type& x, matrix_type& jacobian)
    {
        for (std::size_t i = 0; i < num_residuals; ++i)
        {
            for (std::size_t j = 0; j < num_parameters; ++j)
            {
                jacobian(static_cast<index_type>(i), static_cast<index_type>(j)) =
                    coefficients[i * num_parameters + j] *
                    (std::cos(x[static_cast<index_type>(j)]) + 0.1 * x[static_cast<index_type>(j)]);
            }
        }
    };
    return {std::move(problem), std::move(functor)};
}

void validate_result(const BenchmarkCase& benchmark_case, const api::solver_result& result)
{
    if (!result.converged())
    {
        throw std::runtime_error(benchmark_case.name + " did not converge: " + result.message);
    }
    if (!result.effective_derivative_source ||
        *result.effective_derivative_source != benchmark_case.expected_source)
    {
        throw std::runtime_error(benchmark_case.name + " used an unexpected derivative source");
    }
}

Results measure(Cases& cases)
{
    using clock = std::chrono::steady_clock;

    std::array<std::vector<double>, kNumCases> samples;
    std::array<api::solver_result, kNumCases>  latest;
    for (auto& values : samples)
    {
        values.reserve(kSamples);
    }

    // Exercise every path before timing it. Rotate the starting case so no
    // derivative mode consistently benefits from running after another one.
    for (std::size_t round = 0; round < kWarmupRounds; ++round)
    {
        for (std::size_t position = 0; position < cases.size(); ++position)
        {
            const std::size_t index = (round + position) % cases.size();
            latest[index]           = cases[index].solve();
            validate_result(cases[index], latest[index]);
        }
    }

    for (std::size_t sample = 0; sample < kSamples; ++sample)
    {
        for (std::size_t position = 0; position < cases.size(); ++position)
        {
            const std::size_t index = (sample + position) % cases.size();
            const auto        start = clock::now();
            for (std::size_t run = 0; run < kBatchSize; ++run)
            {
                latest[index] = cases[index].solve();
            }
            const auto end = clock::now();

            validate_result(cases[index], latest[index]);
            const double elapsed_us =
                std::chrono::duration<double, std::micro>(end - start).count();
            samples[index].push_back(elapsed_us / static_cast<double>(kBatchSize));
        }
    }

    Results results;
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        auto& values = samples[index];
        std::sort(values.begin(), values.end());

        auto& result             = results[index];
        result.name              = cases[index].name;
        result.median_time_us    = values[values.size() / 2];
        result.iterations        = static_cast<int>(latest[index].iterations);
        result.final_objective   = latest[index].objective;
        result.converged         = latest[index].converged();
        result.derivative_source = api::to_string(
            latest[index].effective_derivative_source.value_or(api::derivative_mode::automatic));
    }
    return results;
}

void benchmark_problem(const std::string&     heading,
    const testing::optimization_test_problem& test_problem,
    api::least_squares_problem                ad_problem,
    std::size_t                               max_iterations)
{
    std::cout << "\n========== " << heading << " ==========\n";

    api::least_squares_problem analytical_problem;
    analytical_problem.num_parameters = test_problem.num_parameters;
    analytical_problem.num_residuals  = test_problem.num_residuals;
    analytical_problem.residuals      = test_problem.residuals;
    analytical_problem.set_jacobian(test_problem.jacobian);

    const vector_type x0 = to_vector_type(test_problem.initial_guess);

    api::solve_options common_options;
    common_options.backend        = api::backend::ceres;
    common_options.max_iterations = max_iterations;

    auto analytical_options        = common_options;
    analytical_options.derivatives = api::derivative_mode::supplied;

    auto ad_options        = common_options;
    ad_options.derivatives = api::derivative_mode::automatic_differentiation;

    auto finite_difference_options        = common_options;
    finite_difference_options.derivatives = api::derivative_mode::finite_difference;

    // Native LM consuming the AD provider: measures the per-call cost of the
    // provider path (cost function built once per solve, not per Jacobian).
    auto native_ad_options    = ad_options;
    native_ad_options.backend = api::backend::native;

    Cases cases{{
        {"Analytical Jacobian",
            api::derivative_mode::supplied,
            [&]() { return api::solve(analytical_problem, x0, analytical_options); }},
        {"Ceres AD (stride=4)",
            api::derivative_mode::automatic_differentiation,
            [&]() { return api::solve(ad_problem, x0, ad_options); }},
        {"Finite Differences",
            api::derivative_mode::finite_difference,
            [&]() { return api::solve(ad_problem, x0, finite_difference_options); }},
        {"Native LM + AD provider",
            api::derivative_mode::automatic_differentiation,
            [&]() { return api::solve(ad_problem, x0, native_ad_options); }},
    }};

    const Results results = measure(cases);
    BenchmarkResult::print_header();
    for (const auto& result : results)
    {
        result.print_row();
    }
}

void benchmark_rosenbrock()
{
    const auto test_problem = testing::make_rosenbrock_problem();
    auto       ad_problem   = least_squares(
        testing::RosenbrocResiduals{}, test_problem.num_parameters, test_problem.num_residuals);
    ad_problem.derivatives(auto_diff());
    benchmark_problem("ROSENBROCK (2D, 2 residuals)", test_problem, std::move(ad_problem), 100);
}

void benchmark_powell_singular()
{
    const auto test_problem = testing::make_powell_singular_problem();
    auto       ad_problem   = least_squares(testing::PowellSingularResiduals{},
        test_problem.num_parameters,
        test_problem.num_residuals);
    ad_problem.derivatives(auto_diff());
    benchmark_problem(
        "POWELL SINGULAR (4D, 4 residuals)", test_problem, std::move(ad_problem), 200);
}

void benchmark_exponential_fit()
{
    const auto test_problem = testing::make_exponential_fit_problem();

    constexpr double    true_a      = 2.0;
    constexpr double    true_b      = -0.3;
    constexpr size_t    num_samples = 10;
    std::vector<double> sample_times(num_samples);
    std::vector<double> sample_values(num_samples);
    for (size_t i = 0; i < num_samples; ++i)
    {
        sample_times[i]  = static_cast<double>(i);
        sample_values[i] = true_a * std::exp(true_b * sample_times[i]);
    }

    auto ad_problem = least_squares(testing::ExponentialFitResiduals{sample_times, sample_values},
        test_problem.num_parameters,
        test_problem.num_residuals);
    ad_problem.derivatives(auto_diff());
    benchmark_problem(
        "EXPONENTIAL FIT (2D, 10 residuals)", test_problem, std::move(ad_problem), 200);
}

void benchmark_dense_nonlinear()
{
    auto [test_problem, functor] = make_dense_nonlinear_problem();
    auto ad_problem =
        least_squares(functor, test_problem.num_parameters, test_problem.num_residuals);
    ad_problem.derivatives(auto_diff());
    benchmark_problem(
        "DENSE NONLINEAR (16D, 128 residuals)", test_problem, std::move(ad_problem), 100);
}

}  // namespace
}  // namespace solverslib

int main()
try
{
    std::cout << "Ceres Automatic Differentiation Benchmark\n"
              << "Comparing analytical, automatic, and numerical Jacobians\n"
              << "Each value is the median of " << solverslib::kSamples << " batches of "
              << solverslib::kBatchSize << " complete solves after warm-up.\n\n";

    solverslib::benchmark_rosenbrock();
    solverslib::benchmark_powell_singular();
    solverslib::benchmark_exponential_fit();
    solverslib::benchmark_dense_nonlinear();

    std::cout << "\nThe small models primarily measure solver and callback overhead; the "
                 "dense case shows how derivative cost scales with parameter count.\n";
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Benchmark error: " << e.what() << '\n';
    return 1;
}

#else

int main()
{
    std::cerr << "Ceres not enabled (SOLVERS_ENABLE_CERES=OFF)\n";
    return 1;
}

#endif
