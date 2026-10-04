#include <cstdio>

#include "detail/eigen_support.h"
#include "solvers/api/solve.h"

int main()
{
    using namespace solverslib;
    using namespace solverslib::api;

    constexpr size_t num_params    = 2;
    constexpr size_t num_residuals = 3;

    double data_x[] = {1.0, 2.0, 3.0};
    double data_y[] = {2.1, 3.9, 6.2};

    least_squares_problem problem;
    problem.num_parameters = num_params;
    problem.num_residuals  = num_residuals;
    problem.residuals      = [&](vector_type const& p, vector_type& r)
    {
        for (size_t i = 0; i < num_residuals; ++i)
            r(static_cast<index_type>(i)) = p(0) * data_x[i] + p(1) - data_y[i];
    };
    // Optional: omit it and the solver differentiates numerically.
    problem.set_jacobian(
        [&](vector_type const& /*p*/, matrix_type& J)
        {
            for (size_t i = 0; i < num_residuals; ++i)
            {
                J(static_cast<index_type>(i), 0) = data_x[i];
                J(static_cast<index_type>(i), 1) = 1.0;
            }
        });

    vector_type initial_guess(num_params);
    initial_guess << 1.0, 0.0;

    solve_options options;
    options.max_iterations      = 100;
    options.function_tolerance  = 1e-8;
    options.parameter_tolerance = 1e-8;

    const solver_result result = solve(problem, initial_guess, options);

    std::printf(
        "LM: a = %.6f, b = %.6f  (y = a*x + b)\n", result.parameters(0), result.parameters(1));
    std::printf("    status = %s, iterations = %zu, residual = %.2e\n",
        to_string(result.status),
        result.iterations,
        result.residual_norm.value_or(-1.0));

    return result.converged() ? 0 : 1;
}
