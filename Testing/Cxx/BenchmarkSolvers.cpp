// Cross-backend benchmark using the problem-structure API.
//
// Covers every dispatch path in the README decision tree:
//
//   LS + Jacobian + small   → Native LM / Native GN
//   LS + no Jacobian        → POUNDERS (PETSc/TAO)
//   LS + Jacobian (TAO pin) → TAO BRGN
//   Objective + small       → Native L-BFGS
//   Objective (Ipopt pin)   → Ipopt interior point
//   Objective (TAO pin)     → TAO LMVM
//   LS (Ceres pin)          → Ceres LM
//
// Timing: warmup runs to prime caches, then several timed repeats reported
// as best-of and median (same idea as Eigen's bench/BenchTimer.h).
//
// Build: -DSOLVERS_ENABLE_BENCHMARKS=ON and optionally
//   -DSOLVERS_ENABLE_CERES=ON -DSOLVERS_ENABLE_IPOPT=ON -DSOLVERS_ENABLE_PETSC=ON
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "optimization_test_problems.h"
#include "solvers/api/solve.h"

namespace
{
using clock_type = std::chrono::steady_clock;
using namespace solverslib;
using namespace solverslib::api;

struct timing_result
{
    double best_us;
    double median_us;
};

template <typename Fn> timing_result time_fn(Fn&& fn, int warmup = 3, int runs = 11)
{
    for (int i = 0; i < warmup; ++i)
        fn();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i)
    {
        const auto start = clock_type::now();
        fn();
        const auto end = clock_type::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2]};
}

struct benchmark_row
{
    std::string problem_name;
    std::string solver_name;
    std::string impl;
    std::string dispatch_path;
    bool        available;
    bool        converged;
    double      residual_norm;
    size_t      iterations;
    double      best_us;
    double      median_us;
};

void print_section(const std::string& title)
{
    std::cout << "\n=== " << title << " ===\n\n";
    std::cout << std::left << std::setw(18) << "Problem" << std::setw(14) << "Solver"
              << std::setw(12) << "Backend" << std::setw(10) << "Status" << std::setw(14) << "||r||"
              << std::setw(7) << "Iters" << std::setw(14) << "Best (us)" << std::setw(14)
              << "Median (us)"
              << "\n";
    std::cout << std::string(103, '-') << "\n";
}

void print_row(const benchmark_row& r)
{
    std::cout << std::left << std::setw(18) << r.problem_name << std::setw(14) << r.solver_name
              << std::setw(12) << r.impl;

    if (!r.available)
    {
        std::cout << std::setw(10) << "n/a"
                  << "not compiled in\n";
        return;
    }

    std::cout << std::setw(10) << (r.converged ? "OK" : "FAIL") << std::setw(14) << std::scientific
              << std::setprecision(3) << r.residual_norm << std::setw(7) << std::fixed
              << r.iterations << std::setw(14) << std::fixed << std::setprecision(2) << r.best_us
              << std::setw(14) << r.median_us << "\n";
}

double compute_residual_norm(const testing::optimization_test_problem& tp, const vector_type& x)
{
    vector_type r = make_vector(tp.num_residuals);
    tp.residuals(x, r);
    return r.norm();
}

least_squares_problem make_ls(const testing::optimization_test_problem& tp, bool with_jacobian)
{
    least_squares_problem p;
    p.num_parameters = tp.num_parameters;
    p.num_residuals  = tp.num_residuals;
    p.residuals      = tp.residuals;
    if (with_jacobian)
        p.jacobian = tp.jacobian;
    return p;
}

optimization_problem make_obj(const testing::optimization_test_problem& tp)
{
    optimization_problem p;
    p.num_parameters = tp.num_parameters;
    p.objective      = [&tp](const vector_type& x) -> double
    {
        vector_type r = make_vector(tp.num_residuals);
        tp.residuals(x, r);
        return 0.5 * r.squaredNorm();
    };
    p.gradient = [&tp](const vector_type& x, vector_type& g)
    {
        vector_type r = make_vector(tp.num_residuals);
        matrix_type J = make_matrix(tp.num_residuals, tp.num_parameters);
        tp.residuals(x, r);
        tp.jacobian(x, J);
        g = J.transpose() * r;
    };
    return p;
}

benchmark_row run_ls_benchmark(const testing::optimization_test_problem& tp,
    const std::string&                                                   solver_name,
    const std::string&                                                   impl_name,
    const std::string&                                                   path,
    solve_options                                                        opts,
    bool                                                                 with_jacobian = true)
{
    least_squares_problem ls = make_ls(tp, with_jacobian);
    vector_type           x0 = to_vector_type(tp.initial_guess);

    auto result    = solve(ls, x0, opts);
    bool available = result.status != solver_status::backend_unavailable;

    timing_result timing{0.0, 0.0};
    if (available)
    {
        timing = time_fn([&] { solve(ls, x0, opts); });
        result = solve(ls, x0, opts);
    }

    return {tp.name,
        solver_name,
        impl_name,
        path,
        available,
        result.converged(),
        result.residual_norm.value_or(0.0),
        result.iterations,
        timing.best_us,
        timing.median_us};
}

benchmark_row run_obj_benchmark(const testing::optimization_test_problem& tp,
    const std::string&                                                    solver_name,
    const std::string&                                                    impl_name,
    const std::string&                                                    path,
    solve_options                                                         opts)
{
    optimization_problem obj = make_obj(tp);
    vector_type          x0  = to_vector_type(tp.initial_guess);

    auto result    = solve(obj, x0, opts);
    bool available = result.status != solver_status::backend_unavailable;

    timing_result timing{0.0, 0.0};
    double        res_norm = 0.0;
    if (available)
    {
        timing = time_fn([&] { solve(obj, x0, opts); });
        result = solve(obj, x0, opts);
        if (result.has_usable_iterate())
            res_norm = compute_residual_norm(tp, result.parameters);
    }

    return {tp.name,
        solver_name,
        impl_name,
        path,
        available,
        result.converged(),
        res_norm,
        result.iterations,
        timing.best_us,
        timing.median_us};
}

}  // namespace

int main()
{
    using namespace solverslib;

    const auto problems = testing::make_all_test_problems();

    std::cout << "Solvers Cross-Backend Benchmark\n";
    std::cout << "===============================\n";
    std::cout << "Covers every dispatch path in the README decision tree.\n";

    // --- Path 1: LS + Jacobian + small → Native LM ---
    print_section("LS + Jacobian (small) -> Native LM");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.algorithm           = algorithm::levenberg_marquardt;
        opts.backend             = backend::native;
        opts.max_iterations      = 500;
        opts.function_tolerance  = 1e-14;
        opts.parameter_tolerance = 1e-14;
        print_row(run_ls_benchmark(tp, "LM", "Native", "LS+Jac->LM", opts));
    }

    // --- Path 1b: LS + Jacobian + small → Native GN (explicit pin) ---
    print_section("LS + Jacobian (small) -> Native GN [explicit pin]");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.algorithm           = algorithm::gauss_newton;
        opts.backend             = backend::native;
        opts.max_iterations      = 500;
        opts.function_tolerance  = 1e-14;
        opts.parameter_tolerance = 1e-14;
        print_row(run_ls_benchmark(tp, "GN", "Native", "LS+Jac->GN", opts));
    }

    // --- Path 2: LS + no Jacobian → POUNDERS (PETSc/TAO) ---
    print_section("LS + no Jacobian -> POUNDERS (PETSc/TAO)");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.algorithm      = algorithm::pounders;
        opts.backend        = backend::pounders;
        opts.max_iterations = 500;
        opts.petsc_tao      = petsc_tao_options{.gatol = 1e-10, .grtol = 1e-10};
        print_row(run_ls_benchmark(tp, "POUNDERS", "PETSc/TAO", "LS-noJac->POUNDERS", opts, false));
    }

    // --- Path 3: LS + Jacobian + large → TAO (pinned to BRGN) ---
    print_section("LS + Jacobian (large-scale path) -> TAO BRGN [pinned]");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.backend        = backend::petsc_tao;
        opts.max_iterations = 500;
        opts.petsc_tao =
            petsc_tao_options{.algorithm = tao_algorithm::brgn, .gatol = 1e-10, .grtol = 1e-10};
        print_row(run_ls_benchmark(tp, "TAO-BRGN", "PETSc/TAO", "LS+Jac->TAO", opts));
    }

    // --- Path 4: Objective + no constraints + small → Native L-BFGS ---
    print_section("Objective + no constraints (small) -> Native L-BFGS");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.algorithm           = algorithm::lbfgs;
        opts.backend             = backend::native;
        opts.max_iterations      = 500;
        opts.function_tolerance  = 1e-14;
        opts.parameter_tolerance = 1e-14;
        print_row(run_obj_benchmark(tp, "L-BFGS", "Native", "Obj->LBFGS", opts));
    }

    // --- Path 5: Objective + constraints → Ipopt ---
    print_section("Objective + constraints -> Ipopt");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.backend        = backend::ipopt;
        opts.max_iterations = 500;
        opts.ipopt          = ipopt_options{.tol = 1e-14};
        print_row(run_obj_benchmark(tp, "Ipopt", "Ipopt", "Obj+constr->Ipopt", opts));
    }

    // --- Path 6: Objective + no constraints + large → TAO LMVM ---
    print_section("Objective + no constraints (large-scale path) -> TAO LMVM [pinned]");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.backend        = backend::petsc_tao;
        opts.max_iterations = 500;
        opts.petsc_tao =
            petsc_tao_options{.algorithm = tao_algorithm::lmvm, .gatol = 1e-10, .grtol = 1e-10};
        print_row(run_obj_benchmark(tp, "TAO-LMVM", "PETSc/TAO", "Obj->TAO", opts));
    }

    // --- Ceres (explicit pin, comparison) ---
    print_section("LS + Jacobian -> Ceres [explicit pin]");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.backend             = backend::ceres;
        opts.max_iterations      = 500;
        opts.function_tolerance  = 1e-14;
        opts.parameter_tolerance = 1e-14;
        print_row(run_ls_benchmark(tp, "Ceres-LM", "Ceres", "LS->Ceres", opts));
    }

    // --- Ceres jacobian configurations comparison ---
    print_section("Ceres Jacobian Strategies: Internal vs Provided vs None");
    for (const auto& tp : problems)
    {
        solve_options opts;
        opts.backend             = backend::ceres;
        opts.max_iterations      = 500;
        opts.function_tolerance  = 1e-14;
        opts.parameter_tolerance = 1e-14;

        // Configuration 1: With provided Jacobian (analytical)
        print_row(run_ls_benchmark(tp, "Ceres-LM", "Ceres", "LS+Jac(analytical)", opts, true));

        // Configuration 2: Without Jacobian (finite differences)
        print_row(run_ls_benchmark(tp, "Ceres-LM", "Ceres", "LS+Jac(FD)", opts, false));

        // Note: Configuration 3 (Automatic Differentiation via templated functors)
        // would require extending the API to accept templated residuals.
        // The templated functors are available in optimization_test_problems.h
        // and validated in AutomaticDifferentiation tests.
    }

    std::cout << "\n";
    return 0;
}
