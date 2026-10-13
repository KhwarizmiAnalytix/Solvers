// Convergence comparison of every backend on the Rosenbrock problem:
// iterations, final residual/gradient norm, and wall-clock time, side by
// side for native and third-party backends. Unlike BenchmarkNative.cpp and
// BenchmarkExternal.cpp (which time a steady-state loop via Google
// Benchmark), this prints a single solve's convergence diagnostics from
// api::solver_result.
//
// Run with: ./ConvergenceDemo

#include <chrono>
#include <cstdio>
#include <optional>
#include <string>

#include "api/dispatch.h"
#include "problems/rosenbrock.h"
#include "solver_options/solver_options_bfgs.h"
#include "solver_options/solver_options_ceres.h"
#include "solver_options/solver_options_gn.h"
#include "solver_options/solver_options_ipopt.h"
#include "solver_options/solver_options_lm.h"
#include "solver_options/solver_options_nlopt.h"
#include "solver_options/solver_options_petsc.h"

#if SOLVERS_HAS_CERES
#include "solvers/ceres_solver.h"
#endif
#if SOLVERS_HAS_IPOPT
#include "solvers/ipopt_solver.h"
#endif
#if SOLVERS_HAS_NLOPT
#include "solvers/nlopt_solver.h"
#endif
#if SOLVERS_HAS_PETSC
#include "solvers/petsc_tao_solver.h"
#endif

using namespace solverslib;
using namespace solverslib::api;
using namespace solverslib::test;

namespace
{
const char* status_name(solver_status s)
{
    switch (s)
    {
        case solver_status::converged: return "converged";
        case solver_status::max_iterations: return "max_iterations";
        case solver_status::invalid_problem: return "invalid_problem";
        case solver_status::numerical_failure: return "numerical_failure";
        case solver_status::infeasible: return "infeasible";
        case solver_status::unsupported_capability: return "unsupported";
        case solver_status::backend_unavailable: return "unavailable";
        case solver_status::user_stopped: return "user_stopped";
        case solver_status::stalled: return "stalled";
    }
    return "?";
}

void print_least_squares_header()
{
    std::printf(
        "%-28s %-10s %10s %14s %12s %14s\n", "Backend / algorithm", "status", "iterations",
        "||r(x)||", "f(x)", "time (us)");
    std::printf(
        "---------------------------- ---------- ---------- -------------- "
        "------------ --------------\n");
}

void print_optimization_header()
{
    std::printf(
        "%-28s %-10s %10s %14s %12s %14s\n", "Backend / algorithm", "status", "iterations",
        "||grad||", "f(x)", "time (us)");
    std::printf(
        "---------------------------- ---------- ---------- -------------- "
        "------------ --------------\n");
}

// adapter_reports_iterations: Ceres and NLopt never populate solver_result::iterations
// (see src/api/dispatch.cpp), so a bare 0 from those two backends is not a
// measurement. Native, Ipopt, and PETSc/TAO always fill it.
void format_iterations(char* buf, std::size_t buf_size, std::size_t iterations, bool reported)
{
    if (reported)
    {
        std::snprintf(buf, buf_size, "%zu", iterations);
    }
    else
    {
        std::snprintf(buf, buf_size, "n/a");
    }
}

void format_optional(char* buf, std::size_t buf_size, std::optional<double> value)
{
    if (value)
    {
        std::snprintf(buf, buf_size, "%.6e", *value);
    }
    else
    {
        std::snprintf(buf, buf_size, "n/a");
    }
}

// Least-squares form: F(x) = 0.5||r(x)||^2 (or ||r(x)||^2 for native LM/GN; see
// README "Results and current limits"). Reports ||r(x)||, not ||grad||.
void run_and_print_least_squares(const std::string& label, const least_squares_problem& problem,
    const vector_type& x0, const solver_options& options, bool adapter_reports_iterations = true)
{
    const auto start  = std::chrono::steady_clock::now();
    const auto result = solve(problem, x0, options);
    const auto stop   = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(stop - start).count();

    char iterations_buf[16];
    format_iterations(iterations_buf, sizeof(iterations_buf), result.iterations,
        adapter_reports_iterations);
    char residual_buf[16];
    format_optional(residual_buf, sizeof(residual_buf), result.residual_norm);

    std::printf("%-28s %-10s %10s %14s %12.6e %14.2f\n", label.c_str(),
        status_name(result.status), iterations_buf, residual_buf, result.objective, us);
}

// General objective form: min f(x). Reports ||grad||, not ||r(x)||.
void run_and_print_optimization(const std::string& label, const optimization_problem& problem,
    const vector_type& x0, const solver_options& options, bool adapter_reports_iterations = true)
{
    const auto start  = std::chrono::steady_clock::now();
    const auto result = solve(problem, x0, options);
    const auto stop   = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(stop - start).count();

    char iterations_buf[16];
    format_iterations(iterations_buf, sizeof(iterations_buf), result.iterations,
        adapter_reports_iterations);
    char gradient_buf[16];
    format_optional(gradient_buf, sizeof(gradient_buf), result.gradient_norm);

    std::printf("%-28s %-10s %10s %14s %12.6e %14.2f\n", label.c_str(),
        status_name(result.status), iterations_buf, gradient_buf, result.objective, us);
}
}  // namespace

int main()
{
    const auto x0  = rosenbrock_start();
    const auto ls  = rosenbrock_least_squares(true);
    const auto opt = rosenbrock_optimization(true);

    std::printf("Rosenbrock problem, x0 = (%.2f, %.2f), minimum at (1, 1), f(x*) = 0\n\n",
        x0(0), x0(1));

    // Shared option objects: each problem form is solved independently below,
    // never mixing least-squares and general-objective results in one table.
    const auto lm_options = solver_options_lm_builder()
                                 .with_max_iterations(400)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-14)
                                 .with_type(levenberg_marquardt_solver_enum::NIELSEN)
                                 .build();

    const auto gn_options = solver_options_gn_builder()
                                 .with_max_iterations(200)
                                 .with_function_tolerance(1e-14)
                                 .with_gradient_tolerance(1e-12)
                                 .with_parameter_tolerance(1e-14)
                                 .build();

    const auto bfgs_options = solver_options_bfgs_builder()
                                   .with_max_iterations(1000)
                                   .with_function_tolerance(1e-14)
                                   .with_gradient_tolerance(1e-10)
                                   .build();

#if SOLVERS_HAS_CERES
    const auto ceres_options = solver_options_ceres_builder().with_max_iterations(200).build();
#endif
#if SOLVERS_HAS_IPOPT
    const auto ipopt_options = solver_options_ipopt_builder().with_max_iterations(500).build();
#endif
#if SOLVERS_HAS_NLOPT
    const auto nlopt_options = solver_options_nlopt_builder()
                                    .with_max_iterations(500)
                                    .with_algorithm(nlopt_algorithm_enum::LD_LBFGS)
                                    .build();
#endif
#if SOLVERS_HAS_PETSC
    const auto petsc_brgn = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::BRGN)
                                 .with_max_iterations(200)
                                 .build();
    const auto petsc_lmvm = solver_options_petsc_builder()
                                 .with_tao_type(tao_algorithm_enum::LMVM)
                                 .with_max_iterations(500)
                                 .build();
#endif

    // ------------------------------------------------------------------
    // Least-squares form: F(x) = 0.5||r(x)||^2 (or ||r(x)||^2 natively).
    // ------------------------------------------------------------------
    std::printf("Least-squares form (r(x) = [10(x2-x1^2), 1-x1]):\n");
    print_least_squares_header();

    run_and_print_least_squares("native/levenberg_marquardt", ls, x0, *lm_options);
    run_and_print_least_squares("native/gauss_newton", ls, x0, *gn_options);
#if SOLVERS_HAS_CERES
    if (ceres_solver::is_supported())
    {
        run_and_print_least_squares(
            "ceres", ls, x0, *ceres_options, /*adapter_reports_iterations=*/false);
    }
#endif
#if SOLVERS_HAS_IPOPT
    if (ipopt_solver::is_supported())
    {
        run_and_print_least_squares("ipopt", ls, x0, *ipopt_options);
    }
#endif
#if SOLVERS_HAS_PETSC
    if (petsc_tao_solver::is_supported())
    {
        run_and_print_least_squares("petsc_tao/BRGN", ls, x0, *petsc_brgn);
    }
#endif

    std::printf("\n");

    // ------------------------------------------------------------------
    // General objective form: min f(x), f(x) = 0.5((x2-x1^2)^2 + (1-x1)^2).
    // ------------------------------------------------------------------
    std::printf("Optimization form (min f(x)):\n");
    print_optimization_header();

    run_and_print_optimization("native/lbfgs", opt, x0, *bfgs_options);
#if SOLVERS_HAS_IPOPT
    if (ipopt_solver::is_supported())
    {
        run_and_print_optimization("ipopt", opt, x0, *ipopt_options);
    }
#endif
#if SOLVERS_HAS_NLOPT
    if (nlopt_solver::is_supported())
    {
        run_and_print_optimization(
            "nlopt/LD_LBFGS", opt, x0, *nlopt_options, /*adapter_reports_iterations=*/false);
    }
#endif
#if SOLVERS_HAS_PETSC
    if (petsc_tao_solver::is_supported())
    {
        run_and_print_optimization("petsc_tao/LMVM", opt, x0, *petsc_lmvm);
    }
#endif

    return 0;
}
