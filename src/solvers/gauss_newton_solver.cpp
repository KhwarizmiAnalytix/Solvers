#include "gauss_newton_solver.h"

#include <iomanip>
#include <sstream>

#include "detail/support.h"
#include "solver_options/solver_options_gn.h"

namespace solverslib
{
namespace
{
// logging::strings::format only substitutes plain "{}" placeholders (no
// format specifiers), so the width/precision manipulators these diagnostic
// log lines rely on are applied here, once, before handing the resulting
// string off to SOLVERS_LOG_IF (mirrors levenberg_marquardt_solver.cpp).
std::string fmt_iter(size_t value)
{
    std::ostringstream oss;
    oss << std::setw(3) << value;
    return oss.str();
}

std::string fmt_sci(double value, int precision)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

template <typename T> inline double l2_norm(T const& h)
{
    return h.norm();
}
}  // namespace

gauss_newton_solver::gauss_newton_solver(size_t num_parameters,
    size_t                                      num_residuals,
    gauss_newton_solver::function_type          function,
    gauss_newton_solver::jacobian_type          jacobian)
    : function_(std::move(function)), jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters), num_residuals_(num_residuals)
{
}

solver_output gauss_newton_solver::solve(
    vector_type& parameters, const solver_options_gn& options) const
{
    auto jacobian = jacobian_;
    if (jacobian == nullptr)
    {
        auto bump = options.bump();

        jacobian = [this, bump](vector_type const& x, matrix_type& dy_dx)
        {
            auto number_of_parameters = x.size();

            SOLVERS_CHECK(dy_dx.cols() == number_of_parameters);

            auto number_of_targets = dy_dx.rows();

            vector_type y_plus(number_of_targets);
            vector_type y_minus(number_of_targets);

            vector_type x_tmp(number_of_parameters);
            x_tmp = x;

            for (size_t i = 0; i < number_of_parameters; ++i)
            {
                x_tmp[i] += bump;

                function_(x_tmp, y_plus);

                x_tmp[i] -= 2 * bump;
                function_(x_tmp, y_minus);

                for (size_t j = 0; j < y_plus.size(); ++j)
                {
                    dy_dx(j, i) = 0.5 * (y_plus[j] - y_minus[j]) / bump;
                }

                x_tmp[i] = x[i];
            }
        };
    }

    SOLVERS_CHECK(num_parameters_ == parameters.size());

    const auto n        = num_parameters_;
    const auto m        = num_residuals_;
    const auto max_iter = static_cast<size_t>(options.max_num_iterations());

    vector_type y_p(m), y_trial(m);
    matrix_type J(m, n);
    vector_type gradient(n), step(n);

    function_(parameters, y_p);
    auto x2_p                 = l2_norm(y_p);
    auto x2_converged         = x2_p < options.function_tolerance();
    bool gradient_converged   = false;
    bool parameters_converged = false;

    size_t iteration = 0;

    for (; !x2_converged && !parameters_converged && !gradient_converged && iteration < max_iter;
         ++iteration)
    {
        jacobian(parameters, J);
        gradient = J.transpose() * y_p;

        linear_system_solver factorization(J.transpose() * J);
        step = factorization.solve(gradient);

        // step is a descent direction for f(x) = 0.5*||r(x)||^2 whenever J
        // has full column rank: step^T * gradient == step^T * (J^T J) *
        // step >= 0, so directional_derivative <= 0.
        const auto directional_derivative = -step.dot(gradient);

        double step_scale = 1.0;
        bool   accepted   = false;
        double x2_trial   = x2_p;
        vector_type trial(n);

        for (size_t line_search_iter = 0; line_search_iter < options.max_line_search_iterations();
             ++line_search_iter)
        {
            trial = parameters - step_scale * step;
            function_(trial, y_trial);
            x2_trial = l2_norm(y_trial);

            // Armijo sufficient-decrease condition on f(x) = 0.5*||r(x)||^2.
            if (0.5 * x2_trial * x2_trial <=
                0.5 * x2_p * x2_p +
                    options.line_search_sufficient_decrease() * step_scale * directional_derivative)
            {
                accepted = true;
                break;
            }

            step_scale *= options.line_search_backtracking_factor();
        }

        if (!accepted)
        {
            SOLVERS_LOG_IF(INFO,
                options.verbose(),
                "GN Iter {} | LINE SEARCH FAILED | f(x) = {} | best trial = {}",
                fmt_iter(iteration),
                fmt_sci(x2_p, 3),
                fmt_sci(x2_trial, 3));
            break;
        }

        const auto previous_x2 = x2_p;
        parameters              = trial;
        y_p                     = y_trial;
        x2_p                    = x2_trial;

        SOLVERS_LOG_IF(INFO,
            options.verbose(),
            "GN Iter {} | ACCEPTED STEP | f(x) = {} | prev = {} | improvement = {} | step_scale = {}",
            fmt_iter(iteration),
            fmt_sci(x2_p, 3),
            fmt_sci(previous_x2, 3),
            fmt_sci(previous_x2 - x2_p, 2),
            fmt_sci(step_scale, 2));

        const auto step_norm     = l2_norm(step_scale * step);
        const auto param_norm    = l2_norm(parameters);
        const auto gradient_norm = l2_norm(gradient);

        parameters_converged = step_norm < param_norm * options.parameter_tolerance();
        gradient_converged   = gradient_norm < options.gradient_tolerance();
        x2_converged         = x2_p < options.function_tolerance();

        if (parameters_converged || gradient_converged || x2_converged)
        {
            SOLVERS_LOG_IF(INFO,
                options.verbose(),
                "GN Iter {} | CONVERGENCE CHECK | rel_step = {} | grad_norm = {} | f(x) = {}",
                fmt_iter(iteration),
                fmt_sci(step_norm / param_norm, 2),
                fmt_sci(gradient_norm, 2),
                fmt_sci(x2_p, 3));
        }
    }

    SOLVERS_LOG_IF(INFO,
        options.verbose(),
        "GN COMPLETED | iterations = {} | final f(x) = {} | {}{}{}{}",
        iteration,
        fmt_sci(x2_p, 3),
        (x2_converged ? "FUNCTION_CONVERGED " : ""),
        (parameters_converged ? "PARAMETERS_CONVERGED " : ""),
        (gradient_converged ? "GRADIENT_CONVERGED " : ""),
        (iteration >= max_iter ? "MAX_ITERATIONS_REACHED " : ""));

    solver_output output(num_residuals_);
    output.update(x2_converged, parameters_converged, gradient_converged, iteration, y_p);
    return output;
}
}  // namespace solverslib
