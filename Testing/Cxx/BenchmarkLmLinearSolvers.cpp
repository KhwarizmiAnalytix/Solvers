// Iteration counts and final costs of native Levenberg-Marquardt under both
// linear solvers, on the library's standard least-squares problems.
#include <iomanip>
#include <iostream>

#include "optimization_test_problems.h"
#include "solvers/api/solve.h"

int main()
{
    using namespace solverslib;
    using namespace solverslib::api;

    std::cout << std::left << std::setw(16) << "Problem" << std::setw(14) << "Linear solver"
              << std::setw(15) << "Status" << std::setw(8) << "Iters" << std::setw(16)
              << "Final cost F" << "\n"
              << std::string(69, '-') << "\n";
    int failures = 0;
    for (const auto& tp : testing::make_all_test_problems())
    {
        for (const auto solver : {lm_linear_solver::normal_ldlt, lm_linear_solver::augmented_qr})
        {
            least_squares_problem problem;
            problem.num_parameters = tp.num_parameters;
            problem.num_residuals  = tp.num_residuals;
            problem.residuals      = tp.residuals;
            problem.set_jacobian(tp.jacobian);

            solve_options options;
            options.backend             = backend::native;
            options.algorithm           = algorithm::levenberg_marquardt;
            options.max_iterations      = 500;
            options.function_tolerance  = 1e-24;
            options.gradient_tolerance  = 1e-14;
            options.parameter_tolerance = 1e-14;
            options.lm                  = lm_options{};
            options.lm->linear_solver   = solver;

            const auto result = solve(problem, to_vector_type(tp.initial_guess), options);
            std::cout << std::left << std::setw(16) << tp.name << std::setw(14)
                      << (solver == lm_linear_solver::normal_ldlt ? "normal_ldlt" : "augmented_qr")
                      << std::setw(15) << to_string(result.status) << std::setw(8)
                      << result.iterations << std::scientific << std::setprecision(3)
                      << result.objective << "\n";
            failures += result.converged() ? 0 : 1;
        }
    }
    return failures == 0 ? 0 : 1;
}
