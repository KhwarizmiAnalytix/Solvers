// Reproduction cases for solver-redesign-2026-09-26.md.
// Prints observations; these are not assertions of the intended API contract.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_lm.h"
#include "solver_wrapper.h"
#include "solvers/lbfgs_solver.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solvers/polynomial_solver.h"
#include "solvers/root_finding_algorithms.h"

using namespace solverslib;

int main()
{
    std::cout << std::setprecision(17) << std::boolalpha;
    auto residual = [](const vector_type& x, vector_type& r) { r[0] = x[0] - 2.; };
    auto jacobian = [](const vector_type&, matrix_type& j) { j(0, 0) = 1.; };
    solver_wrapper bounded(1, 1, residual, jacobian, {0.}, {1.});
    for (auto options : std::vector<std::shared_ptr<const solver_options>>{
             std::make_shared<solver_options_lm>(100, 1e-12, 1e-12, 1e-12),
             std::make_shared<solver_options_gn>(100, 1e-12, 1e-12, 1e-12),
             std::make_shared<solver_options_bfgs>(100, 1e-12, 1e-12, 1e-12)})
    {
        std::vector<double> x{0.};
        const bool success = bounded.solve(x, options);
        std::cout << "bounds solver=" << static_cast<int>(options->solver())
                  << " success=" << success << " x=" << x[0] << " upper=1\n";
    }

    auto builder = solver_options_lm_builder().with_max_iterations(7);
    auto snapshot = builder.build();
    builder.with_max_iterations(99);
    std::cout << "builder snapshot iterations=" << snapshot->max_num_iterations() << " expected=7\n";

    double root = -999.;
    bool success = root_finding_algorithms::dekker(
        [](double x, double& derivative) { derivative = 1.; return x; },
        -1., 3., root, root_finding_options(50, 1e-12, 1e-12, 2.));
    std::cout << "dekker offset success=" << success << " root=" << root << " expected=2\n";

    success = root_finding_algorithms::bisection(
        [](double) { return 1e-200; }, 0., 1., root,
        root_finding_options(100, 1e-250, 1e-12));
    std::cout << "constant positive bracket success=" << success << " root=" << root << " no root exists\n";

    auto nonzero_residual = [](const vector_type& x, vector_type& r) {
        r[0] = x[0] - 1.; r[1] = x[0] + 1.;
    };
    auto nonzero_jacobian = [](const vector_type&, matrix_type& j) {
        j(0, 0) = 1.; j(1, 0) = 1.;
    };
    vector_type x = vector_type::Zero(1);
    levenberg_marquardt_solver lm(1, 2, nonzero_residual, nonzero_jacobian);
    auto output = lm.solve(x, solver_options_lm(100, 1e-12, 1e-12, 1e-12));
    std::cout << "stationary LM status=" << static_cast<int>(output.status_)
              << " x=" << x[0] << " residual_norm=" << output.x2_ << " expected gradient convergence\n";

    lbfgs_solver bfgs(1, 2, nonzero_residual, nonzero_jacobian);
    solver_options_bfgs stationary_options(100, 1e-12, 1e-12, 1e-12);
    stationary_options.set_type(lbfgs_line_search_type::BACKTRACKING);
    try
    {
        output = bfgs.solve(x, stationary_options);
        std::cout << "stationary BFGS status=" << static_cast<int>(output.status_) << '\n';
    }
    catch (const std::exception&)
    {
        std::cout << "stationary BFGS threw=true expected gradient convergence\n";
    }

    lbfgs_solver steep(1, 1,
        [](const vector_type& p, vector_type& r) { r[0] = 10. * (p[0] - 1.); },
        [](const vector_type&, matrix_type& j) { j(0, 0) = 10.; });
    solver_options_bfgs exhausted(1, 1e-12, 1e-12, 1e-12);
    exhausted.set_type(lbfgs_line_search_type::BACKTRACKING);
    exhausted.set_max_iteration_linesearch(1);
    x[0] = 0.;
    output = steep.solve(x, exhausted);
    std::cout << "exhausted BFGS line search x=" << x[0]
              << " initial_cost=100 final_cost=" << output.x2_ * output.x2_ << '\n';

    std::cout << "quadratic repeated root="
              << polynomial_solver::second_degree_polynomial_solver(-2., 1.) << " expected=1\n";
    std::cout << "quartic min-positive selection="
              << polynomial_solver::fourth_degree_polynomial_solver(0., -10., 0., 9.)
              << " documented minimum=1\n";
    root = polynomial_solver::fourth_degree_polynomial_solver(0., -10., 0., 9., 10.);
    std::cout << "quartic clamped root=" << root
              << " residual=" << root * root * root * root - 10. * root * root + 9. << '\n';
    root = polynomial_solver::fourth_degree_polynomial_solver(0., 0., 0., 1.);
    std::cout << "quartic without real roots=" << root
              << " residual=" << root * root * root * root + 1. << '\n';
}
