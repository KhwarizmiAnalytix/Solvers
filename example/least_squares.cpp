#include <cstdio>

#include "detail/eigen_support.h"
#include "solvers/levenberg_marquardt_solver.h"
#include "solver_options/solver_options_lm.h"

int main()
{
    using solverslib::vector_type;
    using solverslib::matrix_type;

    constexpr size_t num_params   = 2;
    constexpr size_t num_residuals = 3;

    double data_x[] = {1.0, 2.0, 3.0};
    double data_y[] = {2.1, 3.9, 6.2};

    auto residuals = [&](vector_type const& p, vector_type& r) {
        for (size_t i = 0; i < num_residuals; ++i)
            r(static_cast<Eigen::Index>(i)) =
                p(0) * data_x[i] + p(1) - data_y[i];
    };

    auto jacobian = [&](vector_type const& /*p*/, matrix_type& J) {
        for (size_t i = 0; i < num_residuals; ++i) {
            J(static_cast<Eigen::Index>(i), 0) = data_x[i];
            J(static_cast<Eigen::Index>(i), 1) = 1.0;
        }
    };

    solverslib::levenberg_marquardt_solver solver(
        num_params, num_residuals, residuals, jacobian);

    vector_type params(num_params);
    params << 1.0, 0.0;

    auto options = solverslib::solver_options_lm_builder()
        .with_max_iterations(100)
        .with_function_tolerance(1e-8)
        .with_parameter_tolerance(1e-8)
        .build();
    auto result = solver.solve(params, *options);

    std::printf("LM: a = %.6f, b = %.6f  (y = a*x + b)\n",
        params(0), params(1));
    std::printf("    converged = %s, iterations = %zu, residual = %.2e\n",
        result.converged() ? "true" : "false",
        result.iterations, result.residual_norm);

    return result.converged() ? 0 : 1;
}
