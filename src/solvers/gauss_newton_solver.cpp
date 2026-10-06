#include "solvers/gauss_newton_solver.h"

#include <iomanip>
#include <optional>
#include <sstream>

#include "detail/native_evaluation.h"
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
    function_type                               function,
    jacobian_type                               jacobian)
    : function_(std::move(function)), jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters), num_residuals_(num_residuals)
{
}

native_result gauss_newton_solver::solve(
    vector_type& parameters, const solver_options_gn& options) const
{
    SOLVERS_CHECK(num_parameters_ == parameters.size());

    std::string failure;  // non-empty once an evaluation fails fatally

    const auto n        = num_parameters_;
    const auto m        = num_residuals_;
    const auto max_iter = static_cast<size_t>(options.max_num_iterations());

    vector_type y_p(m), y_trial(m);
    matrix_type J(m, n);
    vector_type gradient(n), step(n);

    // Compute initial residuals
    try
    {
        function_(parameters, y_p);
        if (!y_p.allFinite())
        {
            failure = "non-finite residuals at the initial point";
        }
    }
    catch (const std::exception& e)
    {
        failure = std::string("exception during residual evaluation: ") + e.what();
    }
    catch (...)
    {
        failure = "unknown exception during residual evaluation";
    }

    auto x2_p                 = l2_norm(y_p);
    auto x2_converged         = failure.empty() && 0.5 * x2_p * x2_p < options.function_tolerance();
    bool gradient_converged   = false;
    bool parameters_converged = false;

    size_t iteration = 0;

    size_t                accepted_steps = 0;
    size_t                rejected_steps = 0;
    bool                  stalled        = false;
    std::optional<double> last_step_norm;
    std::optional<double> final_gradient_norm;

    for (; failure.empty() && !x2_converged && !parameters_converged && !gradient_converged &&
           iteration < max_iter;
        ++iteration)
    {
        // Compute Jacobian
        try
        {
            if (jacobian_)
            {
                jacobian_(parameters, J);
            }
            else
            {
                // Finite difference approximation of Jacobian
                const double h = options.bump() * std::max(1.0, parameters.norm() / n);
                vector_type x_trial = parameters;
                vector_type y_trial_temp(m);

                for (size_t j = 0; j < n; ++j)
                {
                    x_trial[j] = parameters[j] + h;
                    function_(x_trial, y_trial_temp);
                    J.col(j) = (y_trial_temp - y_p) / h;
                    x_trial[j] = parameters[j];
                }
            }
            if (!J.allFinite())
            {
                failure = "non-finite Jacobian";
                break;
            }
        }
        catch (const std::exception& e)
        {
            failure = std::string("exception during Jacobian evaluation: ") + e.what();
            break;
        }
        catch (...)
        {
            failure = "unknown exception during Jacobian evaluation";
            break;
        }
        gradient            = J.transpose() * y_p;
        final_gradient_norm = gradient.norm();

        linear_system_solver factorization(J.transpose() * J);
        step = factorization.solve(gradient);

        // step is a descent direction for f(x) = 0.5*||r(x)||^2 whenever J
        // has full column rank: step^T * gradient == step^T * (J^T J) *
        // step >= 0, so directional_derivative <= 0.
        const auto directional_derivative = -step.dot(gradient);

        double      step_scale = 1.0;
        bool        accepted   = false;
        double      x2_trial   = x2_p;
        vector_type trial(n);

        for (size_t line_search_iter = 0; line_search_iter < options.max_line_search_iterations();
            ++line_search_iter)
        {
            trial = parameters - step_scale * step;
            bool trial_valid = false;

            try
            {
                function_(trial, y_trial);
                trial_valid = y_trial.allFinite();
                if (trial_valid)
                {
                    x2_trial = l2_norm(y_trial);

                    // Armijo sufficient-decrease condition on f(x) = 0.5*||r(x)||^2.
                    // An invalid (non-finite) trial counts as a failed condition.
                    if (0.5 * x2_trial * x2_trial <=
                        0.5 * x2_p * x2_p + options.line_search_sufficient_decrease() * step_scale *
                                                directional_derivative)
                    {
                        accepted = true;
                        break;
                    }
                }
            }
            catch (const std::exception& e)
            {
                failure = std::string("exception during line-search: ") + e.what();
                break;
            }
            catch (...)
            {
                failure = "unknown exception during line-search";
                break;
            }

            ++rejected_steps;
            step_scale *= options.line_search_backtracking_factor();
        }

        if (!failure.empty())
        {
            break;
        }

        if (!accepted)
        {
            stalled = true;
            SOLVERS_LOG_IF(INFO,
                options.verbose(),
                "GN Iter {} | LINE SEARCH FAILED | f(x) = {} | best trial = {}",
                fmt_iter(iteration),
                fmt_sci(x2_p, 3),
                fmt_sci(x2_trial, 3));
            break;
        }

        ++accepted_steps;
        const auto previous_x2 = x2_p;
        parameters             = trial;
        y_p                    = y_trial;
        x2_p                   = x2_trial;

        SOLVERS_LOG_IF(INFO,
            options.verbose(),
            "GN Iter {} | ACCEPTED STEP | f(x) = {} | prev = {} | improvement = {} | step_scale = "
            "{}",
            fmt_iter(iteration),
            fmt_sci(x2_p, 3),
            fmt_sci(previous_x2, 3),
            fmt_sci(previous_x2 - x2_p, 2),
            fmt_sci(step_scale, 2));

        const auto step_norm     = l2_norm(step_scale * step);
        last_step_norm           = step_norm;
        const auto param_norm    = l2_norm(parameters);
        const auto gradient_norm = l2_norm(gradient);

        parameters_converged = step_norm < param_norm * options.parameter_tolerance();
        gradient_converged   = gradient_norm < options.gradient_tolerance();
        x2_converged         = 0.5 * x2_p * x2_p < options.function_tolerance();

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

    native_result result;
    result.iterations    = iteration;
    result.residual_norm = l2_norm(y_p);
    if (!failure.empty())
    {
        result.status  = native_convergence::numerical_failure;
        result.message = failure;
    }
    else if (x2_converged)
    {
        result.status = native_convergence::function_converged;
    }
    else if (parameters_converged)
    {
        result.status = native_convergence::parameter_converged;
    }
    else if (gradient_converged)
    {
        result.status = native_convergence::gradient_converged;
    }
    else if (stalled)
    {
        result.status  = native_convergence::stalled;
        result.message = "line search failed to find a sufficient decrease";
    }
    result.accepted_steps = accepted_steps;
    result.rejected_steps = rejected_steps;
    result.step_norm      = last_step_norm;
    result.gradient_norm  = final_gradient_norm;

    return result;
}
}  // namespace solverslib
