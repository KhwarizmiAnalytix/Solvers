#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace solverslib::testing
{
// Shared least-squares test problems used by both the correctness tests
// (TestSolverBackends.cpp) and the cross-backend benchmark
// (BenchmarkSolvers.cpp), so every backend is measured against the same
// objectives. All problems are posed as residual vectors r(x); every
// backend (LM, LBFGS, NLopt, Ceres) minimizes 0.5 * ||r(x)||^2.
struct optimization_test_problem
{
    using residual_function_type = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using jacobian_function_type = std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)>;

    std::string            name;
    size_t                 num_parameters;
    size_t                 num_residuals;
    std::vector<double>    initial_guess;
    std::vector<double>    expected_solution;
    residual_function_type residuals;
    jacobian_function_type jacobian;
};

// Scalar affine residual: r(x) = x - 2. The simplest possible sanity check.
inline optimization_test_problem make_linear_scalar_problem()
{
    return optimization_test_problem{"LinearScalar",
        1,
        1,
        {0.0},
        {2.0},
        [](const Eigen::VectorXd& x, Eigen::VectorXd& r) { r[0] = x[0] - 2.0; },
        [](const Eigen::VectorXd&, Eigen::MatrixXd& j) { j(0, 0) = 1.0; }};
}

// Rosenbrock's "banana" function in least-squares residual form:
//   r0 = 10 * (x1 - x0^2), r1 = 1 - x0
// Classic quasi-Newton / trust-region benchmark; minimum at (1, 1).
inline optimization_test_problem make_rosenbrock_problem()
{
    return optimization_test_problem{"Rosenbrock2D",
        2,
        2,
        {-1.2, 1.0},
        {1.0, 1.0},
        [](const Eigen::VectorXd& x, Eigen::VectorXd& r)
        {
            r[0] = 10.0 * (x[1] - x[0] * x[0]);
            r[1] = 1.0 - x[0];
        },
        [](const Eigen::VectorXd& x, Eigen::MatrixXd& j)
        {
            j(0, 0) = -20.0 * x[0];
            j(0, 1) = 10.0;
            j(1, 0) = -1.0;
            j(1, 1) = 0.0;
        }};
}

// Powell's singular function: four residuals, four parameters, Jacobian is
// singular at the minimum x = 0. Used by Ceres' own examples/powell.cc to
// stress-test convergence near degenerate curvature.
inline optimization_test_problem make_powell_singular_problem()
{
    return optimization_test_problem{"PowellSingular",
        4,
        4,
        {3.0, -1.0, 0.0, 1.0},
        {0.0, 0.0, 0.0, 0.0},
        [](const Eigen::VectorXd& x, Eigen::VectorXd& r)
        {
            r[0] = x[0] + 10.0 * x[1];
            r[1] = std::sqrt(5.0) * (x[2] - x[3]);
            r[2] = (x[1] - 2.0 * x[2]) * (x[1] - 2.0 * x[2]);
            r[3] = std::sqrt(10.0) * (x[0] - x[3]) * (x[0] - x[3]);
        },
        [](const Eigen::VectorXd& x, Eigen::MatrixXd& j)
        {
            j.setZero();
            j(0, 0) = 1.0;
            j(0, 1) = 10.0;
            j(1, 2) = std::sqrt(5.0);
            j(1, 3) = -std::sqrt(5.0);
            j(2, 1) = 2.0 * (x[1] - 2.0 * x[2]);
            j(2, 2) = -4.0 * (x[1] - 2.0 * x[2]);
            j(3, 0) = 2.0 * std::sqrt(10.0) * (x[0] - x[3]);
            j(3, 3) = -2.0 * std::sqrt(10.0) * (x[0] - x[3]);
        }};
}

// Exponential curve fit y = a * exp(b * t) against noise-free samples
// generated from a = 2.0, b = -0.3, so the true minimum is known exactly.
inline optimization_test_problem make_exponential_fit_problem()
{
    static constexpr double true_a      = 2.0;
    static constexpr double true_b      = -0.3;
    static constexpr size_t num_samples = 10;
    std::vector<double>     sample_times(num_samples);
    std::vector<double>     sample_values(num_samples);
    for (size_t i = 0; i < num_samples; ++i)
    {
        sample_times[i]  = static_cast<double>(i);
        sample_values[i] = true_a * std::exp(true_b * sample_times[i]);
    }

    return optimization_test_problem{"ExponentialFit",
        2,
        num_samples,
        {1.0, 0.0},
        {true_a, true_b},
        [sample_times, sample_values](const Eigen::VectorXd& x, Eigen::VectorXd& r)
        {
            for (size_t i = 0; i < sample_times.size(); ++i)
            {
                r[static_cast<Eigen::Index>(i)] =
                    x[0] * std::exp(x[1] * sample_times[i]) - sample_values[i];
            }
        },
        [sample_times](const Eigen::VectorXd& x, Eigen::MatrixXd& j)
        {
            for (size_t i = 0; i < sample_times.size(); ++i)
            {
                const double t                     = sample_times[i];
                const double exp_bt                = std::exp(x[1] * t);
                j(static_cast<Eigen::Index>(i), 0) = exp_bt;
                j(static_cast<Eigen::Index>(i), 1) = x[0] * t * exp_bt;
            }
        }};
}

inline std::vector<optimization_test_problem> make_all_test_problems()
{
    return {make_linear_scalar_problem(),
        make_rosenbrock_problem(),
        make_powell_singular_problem(),
        make_exponential_fit_problem()};
}

}  // namespace solverslib::testing
