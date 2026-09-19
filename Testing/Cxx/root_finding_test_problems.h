#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace solverslib::testing
{
// Shared scalar root-finding problems used by both the correctness tests
// (TestRootFindingAndPolynomialSolvers.cpp) and the cross-algorithm benchmark
// (BenchmarkRootFinders.cpp). Every problem supplies both a value-only
// residual (for the derivative-free methods: bisection, false_position,
// ridders, brent, secant) and a value+derivative residual (for dekker and
// newton_raphson), so every root_finding_algorithms method can be measured
// against the same objectives.
struct root_finding_test_problem
{
    using function_type          = std::function<double(double)>;
    using function_gradient_type = std::function<double(double, double&)>;

    std::string            name;
    double                 x1;  // bracket lower bound / first open-method guess
    double                 x2;  // bracket upper bound / second open-method guess
    double                 expected_root;
    function_type          residual;
    function_gradient_type residual_with_derivative;
};

// f(x) = x^2 - 4, root at x = 2. Bracket [0, 3].
inline root_finding_test_problem make_quadratic_root_problem()
{
    return root_finding_test_problem{"Quadratic",
        0.0,
        3.0,
        2.0,
        [](double x) { return x * x - 4.0; },
        [](double x, double& df_dx) {
            df_dx = 2.0 * x;
            return x * x - 4.0;
        }};
}

// f(x) = x^3 - x - 2, root ~ 1.5213797068. Bracket [1, 2].
inline root_finding_test_problem make_cubic_root_problem()
{
    return root_finding_test_problem{"Cubic",
        1.0,
        2.0,
        1.5213797068,
        [](double x) { return x * x * x - x - 2.0; },
        [](double x, double& df_dx) {
            df_dx = 3.0 * x * x - 1.0;
            return x * x * x - x - 2.0;
        }};
}

// f(x) = cos(x) - x, the classic fixed-point benchmark, root ~ 0.7390851332.
// Bracket [0, 1].
inline root_finding_test_problem make_transcendental_root_problem()
{
    return root_finding_test_problem{"Transcendental",
        0.0,
        1.0,
        0.7390851332,
        [](double x) { return std::cos(x) - x; },
        [](double x, double& df_dx) {
            df_dx = -std::sin(x) - 1.0;
            return std::cos(x) - x;
        }};
}

// f(x) = x^3 - 2x - 5, root ~ 2.0945514815. Bracket [2, 3].
inline root_finding_test_problem make_asymmetric_cubic_root_problem()
{
    return root_finding_test_problem{"AsymmetricCubic",
        2.0,
        3.0,
        2.0945514815,
        [](double x) { return x * x * x - 2.0 * x - 5.0; },
        [](double x, double& df_dx) {
            df_dx = 3.0 * x * x - 2.0;
            return x * x * x - 2.0 * x - 5.0;
        }};
}

inline std::vector<root_finding_test_problem> make_all_root_finding_test_problems()
{
    return {make_quadratic_root_problem(),
        make_cubic_root_problem(),
        make_transcendental_root_problem(),
        make_asymmetric_cubic_root_problem()};
}

}  // namespace solverslib::testing
