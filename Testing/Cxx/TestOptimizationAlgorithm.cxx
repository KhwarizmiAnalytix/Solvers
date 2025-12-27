#include <cmath>
#include <cstddef>

#include "expressions/expressions.h"
#include "optimization_algorithm/root_finding_algorithms.h"
#include "optimization_algorithm/solver_options/solver_options_bfgs.h"
#include "optimization_algorithm/solver_options/solver_options_lm.h"
#include "optimization_algorithm/solvers/lbfgs_solver.h"
#include "optimization_algorithm/solvers/levenberg_marquardt_solver.h"
#include "optimization_algorithm/solvers/nlopt_solver.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"
#include "quarismaTest.h"

using namespace quarisma;

namespace
{

// Simple quadratic function: f(x) = x^2 + 1
double quadratic_function(const vector<double>& x, vector<double>& grad)
{
    if (!grad.empty())
    {
        grad[0] = 2.0 * x[0];
    }
    return x[0] * x[0] + 1.0;
}

// Simple constraint: g(x) = x - 0.5 <= 0
double constraint_function(const vector<double>& x, vector<double>& grad)
{
    if (!grad.empty())
    {
        grad[0] = 1;
    }
    return x[0] + 0.5;
}

constexpr double tolerance = 1.E-15;

double lbfgs_output(
    const quarisma::lbfgs_solver&        functor,
    quarisma::vector<double>&            p,
    const quarisma::vector<double>&      b,
    const quarisma::solver_options_bfgs& option)
{
    p = 150.;

    functor.solve(p, option).print();

    double diff = hmax(fabs(p - b));
    QUARISMA_LOGF(INFO, " %.4e ", diff);
    return diff;
}

double levenberg_marquardt_solver_output(
    const quarisma::levenberg_marquardt_solver& functor,
    quarisma::vector<double>&                   p,
    const quarisma::vector<double>&             b,
    const quarisma::solver_options_lm&          option)
{
    p = 150;

    functor.solve(p, option);

    double diff = hmax(fabs(p - b));
    QUARISMA_LOGF(INFO, " %.4e ", diff);
    return diff;
}

void test_lbfgs(
    const quarisma::lbfgs_solver&   functor,
    quarisma::solver_options_bfgs&  params,
    quarisma::vector<double>&       p,
    const quarisma::vector<double>& b)
{
    params.set_step_max(10000);
    params.set_step_min(0.);
    params.set_function_tolerance(std::numeric_limits<double>::epsilon());
    params.set_gradient_tolerance(0.);
    params.set_parameter_tolerance(0.);
    params.set_type(quarisma::lbfgs_line_search_type::BACKTRACKING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::ARMIJO);

    auto diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_type(quarisma::lbfgs_line_search_type::BACKTRACKING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::WOLFE);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_type(quarisma::lbfgs_line_search_type::BACKTRACKING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::STRONG_WOLFE);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_type(quarisma::lbfgs_line_search_type::BRACKETING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::ARMIJO);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_type(quarisma::lbfgs_line_search_type::BRACKETING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::WOLFE);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_type(quarisma::lbfgs_line_search_type::BRACKETING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::STRONG_WOLFE);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_gradient_tolerance(1.e-10);
    params.set_function_tolerance(0.);
    params.set_parameter_tolerance(0.);
    params.set_type(quarisma::lbfgs_line_search_type::BRACKETING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::WOLFE);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 4.e-4);

    //-----------------------------------------------------------------------------
    params.set_gradient_tolerance(0.);
    params.set_function_tolerance(0.);
    params.set_parameter_tolerance(0.);
    params.set_type(quarisma::lbfgs_line_search_type::BRACKETING);
    params.set_method_type(quarisma::lbfgs_line_search_method_type::WOLFE);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 2.e-6);

    //-----------------------------------------------------------------------------
    params.set_gradient_tolerance(0.);
    params.set_function_tolerance(0.);
    params.set_parameter_tolerance(1.e-5);
    params.set_type(quarisma::lbfgs_line_search_type::NOCEDAL_WRIGHT);
    diff = lbfgs_output(functor, p, b, params);
    EXPECT_LE(diff, 1.e-4);

    //-----------------------------------------------------------------------------
    params.set_step_max(0.2);
    params.set_step_min(0.19);
    ASSERT_ANY_THROW({ lbfgs_output(functor, p, b, params); });
}

void test_lm(
    const quarisma::levenberg_marquardt_solver& functor,
    quarisma::solver_options_lm&                params,
    quarisma::vector<double>&                   p,
    const quarisma::vector<double>&             b)
{
#ifndef NDEBUG
    params.set_debug(true);
    params.set_log_file("TestOptimizationAlgorithm.log");
#endif  // !NDEBUG

    params.set_use_geodesic(false);
    params.set_accept_uphill_step(true);
    {
        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::LEVENBERG);
        double diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.5e-8);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::QUADRATIC);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 2.e-4);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::NIELSEN);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-8);
    }

    params.set_use_geodesic(true);
    params.set_accept_uphill_step(true);
    {
        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::LEVENBERG);
        double diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-7);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::QUADRATIC);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 2.e-5);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::NIELSEN);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-8);
    }

    params.set_use_geodesic(true);
    params.set_accept_uphill_step(false);
    {
        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::LEVENBERG);
        double diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-8);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::QUADRATIC);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-5);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::NIELSEN);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-7);
    }

    params.set_use_geodesic(false);
    params.set_accept_uphill_step(false);
    {
        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::LEVENBERG);
        double diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.5e-8);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::QUADRATIC);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 2.e-4);

        //-----------------------------------------------------------------------------
        params.set_type(quarisma::levenberg_marquardt_solver_enum::NIELSEN);
        diff = levenberg_marquardt_solver_output(functor, p, b, params);
        EXPECT_LE(diff, 1.e-8);
    }
}

void test_lbfgs()
{
    size_t n = 3;

    quarisma::vector<double> b(n);
    quarisma::vector<double> p(n);

    b = 0.;

    auto func = [b](quarisma::vector<double> const& x, quarisma::vector<double>& y)
    { y = log1p((x - b) * (x - b)); };

    auto jacobi = [b](quarisma::vector<double> const& x, quarisma::matrix<double>& y)
    {
        y = 0.;

        y[0][0] = 2 * (x[0] - b[0]) / (1 + (x[0] - b[0]) * (x[0] - b[0]));
        y[1][1] = 2 * (x[1] - b[1]) / (1 + (x[1] - b[1]) * (x[1] - b[1]));
        y[2][2] = 2 * (x[2] - b[2]) / (1 + (x[2] - b[2]) * (x[2] - b[2]));
    };

    quarisma::solver_options_bfgs params(100);
    {
        quarisma::lbfgs_solver functor(3, 3, func, jacobi);
        test_lbfgs(functor, params, p, b);
    }
    {
        quarisma::lbfgs_solver functor(3, 3, func);
        test_lbfgs(functor, params, p, b);
    }
}

void test_levenberg_marquardt_solver()
{
    size_t n = 3;
    size_t m = 3;

    quarisma::vector<double> b(n);
    quarisma::vector<double> p(n);

    b[0] = 1.3;
    b[1] = -1.;
    b[2] = 3.;

    auto func = [b](quarisma::vector<double> const& x, quarisma::vector<double>& y)
    { y = log1p((x - b) * (x - b)); };
    auto jacobi = [b](quarisma::vector<double> const& x, quarisma::matrix<double>& y)
    {
        y = 0.;

        y[0][0] = 2 * (x[0] - b[0]) / (1 + (x[0] - b[0]) * (x[0] - b[0]));
        y[1][1] = 2 * (x[1] - b[1]) / (1 + (x[1] - b[1]) * (x[1] - b[1]));
        y[2][2] = 2 * (x[2] - b[2]) / (1 + (x[2] - b[2]) * (x[2] - b[2]));
    };

    quarisma::solver_options_lm params(500);
    {
        QUARISMA_LOGF(INFO, " levenberg_marquardt_solver with jacobian");
        quarisma::levenberg_marquardt_solver functor(n, m, func, jacobi);
        test_lm(functor, params, p, b);
    }
    {
        QUARISMA_LOGF(INFO, " levenberg_marquardt_solver without jacobian");
        quarisma::levenberg_marquardt_solver functor(n, m, func);
        test_lm(functor, params, p, b);
    }
}

void test_brent()
{
    double root = 0.;
    double a    = -4.;
    double b    = 0.95;

    auto func = [](double x) { return (x + 3.) * (x - 1.) * (x - 1.); };

    quarisma::root_finding_algorithms::brent(func, a, b, root, 0., tolerance);

    EXPECT_LE(std::fabs(root + 3), tolerance);
    root = 0;
    EXPECT_TRUE(quarisma::root_finding_algorithms::brent(func, b, a, root, 0., tolerance));
    root = 0.;
    EXPECT_TRUE(quarisma::root_finding_algorithms::brent(func, a, b, root, 0., tolerance));
    EXPECT_TRUE(quarisma::root_finding_algorithms::brent(func, root, b, root, 0., tolerance));
    EXPECT_TRUE(quarisma::root_finding_algorithms::brent(func, a, root, root, 0., tolerance));
    root = 0;
    EXPECT_TRUE(quarisma::root_finding_algorithms::brent(func, b, a, root, 0., tolerance, 1.));

    root = 0;
    EXPECT_FALSE(
        quarisma::root_finding_algorithms::brent(func, b, a, root, 0., tolerance, tolerance, 1));
}

void test_dekker()
{
    double root = 0.;
    double a    = -4.;
    double b    = 0.95;

    auto func_2 = [](double x, double& df_dx)
    {
        auto f = expm1((x + 3.)) * (x - 1.) * (x - 1.);
        df_dx  = 2 * (x - 1.) * expm1((x + 3.)) + (x + 3) * (x - 1.) * (x - 1.) * exp((x + 3.));
        return f;
    };

    quarisma::root_finding_algorithms::dekker(func_2, a, b, root, tolerance);

    EXPECT_LE(std::fabs(root + 3), tolerance);

    root = 0.;
    EXPECT_TRUE(quarisma::root_finding_algorithms::dekker(func_2, b, a, root, tolerance));
    root = 0.;
    EXPECT_TRUE(quarisma::root_finding_algorithms::dekker(func_2, a, b, root, tolerance));
    EXPECT_TRUE(quarisma::root_finding_algorithms::dekker(func_2, root, b, root, tolerance));
    EXPECT_TRUE(quarisma::root_finding_algorithms::dekker(func_2, a, root, root, tolerance));

    root = 0.;
    EXPECT_TRUE(quarisma::root_finding_algorithms::dekker(func_2, a, b, root, tolerance, 1.));

    auto func_3 = [](double x, double& df_dx)
    {
        auto f = -expm1(-(x + 3.)) * (x - 1.) * (x - 1.);
        df_dx = -(2 * (x - 1.) * expm1(-(x + 3.)) - (x + 3) * (x - 1.) * (x - 1.) * exp(-(x + 3.)));
        return f;
    };

    root = b;
    EXPECT_TRUE(
        quarisma::root_finding_algorithms::dekker(
            func_3, a, b, root, 10 * tolerance, 10 * tolerance));

    root = -3. + 0.1;
    EXPECT_FALSE(
        quarisma::root_finding_algorithms::dekker(func_3, -3. - 0.1, -3. + 0.1, root, tolerance));

    root = 0;
    EXPECT_TRUE(quarisma::root_finding_algorithms::dekker(func_3, b, a, root, tolerance, 1.));

    root = 0;
    EXPECT_FALSE(
        quarisma::root_finding_algorithms::dekker(func_3, b, a, root, tolerance, tolerance, 1));
}
}  // namespace
QUARISMATEST(Math, OptimizationAlgorithm)
{
    START_LOG_TO_FILE_NAME(OptimizationAlgorithm);
#if 0
    void test_nlopt_solver(nlopt_algo_name_enum algo)
    {  // Initial guess
        std::vector<double> x = {1.0};

        // Lower and upper bounds
        std::vector<double> lb = {-10.0};
        std::vector<double> ub = {10.0};

        // Tolerances
        double x_tol = 1e-4;
        double f_tol = 1e-4;

        auto optimizer = std::make_unique<nlopt_solver>(
            quadratic_function, constraint_function, x, lb, ub, x_tol, f_tol, algo);

        try
        {
            const std::vector<double>& result = (*optimizer)();

            // The optimal solution should be close to 0.5 due to the constraint
            EXPECT_NEAR(result[0], -0.5, x_tol);

            // Check if the result satisfies the constraint
            EXPECT_LE(result[0] + 0.5, x_tol);

            // Check if the objective function value is close to the expected minimum
            vector<double> grad;
            double         min_value = quadratic_function(vector<double>(result), grad);
            EXPECT_NEAR(min_value, 1.25, f_tol);  // f(0.5) = 0.5^2 + 1 = 1.25
        }
        catch (...)
        {
        }
    }
    test_nlopt_solver(nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN);
    test_nlopt_solver(nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN_WITH_EQUALITY_CONSTRAINTS);
    test_nlopt_solver(nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN_WITH_COBYLA);
    test_nlopt_solver(nlopt_algo_name_enum::AUGMENTED_LAGRANGIAN_WITH_BOBYQA);
    test_nlopt_solver(nlopt_algo_name_enum::METHOD_OF_MOVING_ASYMPTOTES);
    test_nlopt_solver(nlopt_algo_name_enum::SEQUENTIAL_LEAST_SQUARES_PROGRAMMING);
    // test_nlopt_solver(nlopt_algo_name_enum::CONSTRAINED_OPTIMIZATION_BY_LINEAR_APPROXIMATIONS);
    //test_nlopt_solver(nlopt_algo_name_enum::BOUND_OPTIMIZATION_BY_QUADRATIC_APPROXIMATION);
    //test_nlopt_solver(nlopt_algo_name_enum::lbfgs_solver);
    //test_nlopt_solver(nlopt_algo_name_enum::IMPROVED_STOCHASTIC_RANKING_EVOLUTION_STRATEGY);
    //test_nlopt_solver(nlopt_algo_name_enum::CONTROLLED_RANDOM_SEARCH_WITH_LOCAL_MUTATION);
    //test_nlopt_solver(nlopt_algo_name_enum::DIVIDING_RECTANGLES);
    //test_nlopt_solver(nlopt_algo_name_enum::PRECONDITIONED_TRUNCATED_NEWTON_METHOD);
    //test_nlopt_solver(nlopt_algo_name_enum::VARIABLE_METRIC_METHOD);
#endif

    test_lbfgs();

    test_levenberg_marquardt_solver();

    test_brent();

    test_dekker();

    END_LOG_TO_FILE_NAME(OptimizationAlgorithm);
    END_TEST();
}
