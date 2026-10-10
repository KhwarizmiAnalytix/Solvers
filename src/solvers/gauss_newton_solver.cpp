#include "solvers/gauss_newton_solver.h"

#include <iomanip>
#include <optional>
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
    function_type                               function,
    jacobian_type                               jacobian,
    double                                      bump,
    finite_difference_scale                     difference_scale)
    : function_(std::move(function)), jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters), num_residuals_(num_residuals), fd_step_(bump),
      fd_scale_(difference_scale)
{
    if (jacobian_)
    {
        return;
    }

    fd_parameters_.resize(static_cast<index_type>(num_parameters_));
    fd_residual_.resize(static_cast<index_type>(num_residuals_));
    fd_base_.resize(static_cast<index_type>(num_residuals_));
    // Forward bump-and-run. absolute: h = step. relative: h = step * |x_j|.
    jacobian_ = [this](vector_type const& x, matrix_type& jacobian_matrix)
    {
        function_(x, fd_base_);
        fd_parameters_ = x;
        for (size_t j = 0; j < num_parameters_; ++j)
        {
            const double h = finite_difference_increment(fd_scale_, fd_step_, x[j]);
            SOLVERS_CHECK(h > 0.0, "finite-difference step is not positive");
            fd_parameters_[j] = x[j] + h;
            function_(fd_parameters_, fd_residual_);
            jacobian_matrix.col(static_cast<index_type>(j)) = (fd_residual_ - fd_base_) / h;
            fd_parameters_[j]                               = x[j];
        }
    };
}

native_result gauss_newton_solver::solve(
    vector_type& parameters, const solver_options_gn& options) const
{
    SOLVERS_CHECK(num_parameters_ == parameters.size());

    const auto n        = num_parameters_;
    const auto m        = num_residuals_;
    const auto max_iter = static_cast<size_t>(options.max_num_iterations());

    vector_type y_p(m), y_trial(m);
    matrix_type J(m, n);
    vector_type gradient(n), step(n);

    function_(parameters, y_p);
    SOLVERS_CHECK(y_p.allFinite(), "non-finite residuals at the initial point");

    auto x2_p                 = l2_norm(y_p);
    auto x2_converged         = 0.5 * x2_p * x2_p < options.function_tolerance();
    bool gradient_converged   = false;
    bool parameters_converged = false;

    size_t iteration = 0;

    size_t                accepted_steps = 0;
    size_t                rejected_steps = 0;
    bool                  stalled        = false;
    std::optional<double> last_step_norm;
    std::optional<double> final_gradient_norm;

    for (; !x2_converged && !parameters_converged && !gradient_converged && iteration < max_iter;
        ++iteration)
    {
        jacobian_(parameters, J);
        SOLVERS_CHECK(J.allFinite(), "non-finite Jacobian");
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
            trial            = parameters - step_scale * step;
            bool trial_valid = false;

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

            ++rejected_steps;
            step_scale *= options.line_search_backtracking_factor();
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
    if (x2_converged)
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
