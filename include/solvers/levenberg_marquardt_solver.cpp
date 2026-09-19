#include "levenberg_marquardt_solver.h"

#include <iomanip>

#include "detail/support.h"
#include "solver_options/solver_options_lm.h"

#define MAX_LAMBDA 1e12

namespace solverslib
{
template <typename T>
inline double l2_norm(T const& h)
{
    return h.norm();
}

levenberg_marquardt_solver::levenberg_marquardt_solver(
    size_t                                    num_parameters,
    size_t                                    num_residuals,
    levenberg_marquardt_solver::function_type function,
    levenberg_marquardt_solver::jacobian_type jacobian)
    : function_(std::move(function)),
      jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters),
      num_residuals_(num_residuals)
{
}

solver_output levenberg_marquardt_solver::solve(
    Eigen::VectorXd& parameters, const solver_options_lm& options) const
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

    const auto n = num_parameters_;
    const auto m = num_residuals_;

    const auto max_iter           = options.max_num_iterations();
    const auto accept_uphill_step = options.accept_uphill_step();
    const auto alpha              = options.alpha();
    const auto epsilon            = options.epsilon();

    const auto Dh = 2. / (epsilon * epsilon);

    auto lambda = options.lambda();
    auto nu     = options.nu();

    vector_type y_p(m), y_p_new(m), y_tmp(m);
    vector_type JtWdy(n), p_new(n), last_accepted_step = vector_type::Zero(n);
    vector_type step(n), tmp(n), diagonals = vector_type::Ones(n);
    matrix_type J(m, n), Jt(n, m), JtWJ(n, n), JtWJ_lambda(n, n);

    function_(parameters, y_p);
    auto x2_p                 = l2_norm(y_p);
    auto x2_converged         = x2_p < options.function_tolerance();
    bool gradient_converged   = false;
    bool parameters_converged = false;

    size_t iteration = 0;

    if (!x2_converged)
    {
        bool stop = false;
        jacobian(parameters, J);

        Jt    = J.transpose();
        JtWJ  = Jt * J;
        JtWdy = Jt * y_p;

        auto min_x2 = x2_p;

        for (; !stop && iteration < max_iter; ++iteration)
        {
            SOLVERS_CHECK_FINITE_DEBUG(lambda);
            SOLVERS_CHECK_FINITE_DEBUG(nu);

            JtWJ_lambda = JtWJ;
            switch (options.type())
            {
            case levenberg_marquardt_solver_enum::LEVENBERG:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = JtWJ(i, i);
                    JtWJ_lambda(i, i) *= (1 + lambda);
                }
                break;
            }
            case levenberg_marquardt_solver_enum::QUADRATIC:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    JtWJ_lambda(i, i) += lambda;
                }
                break;
            }
            case levenberg_marquardt_solver_enum::NIELSEN:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = std::max(JtWJ(i, i), diagonals[i]);
                    JtWJ_lambda(i, i) += lambda * diagonals[i];
                }
                break;
            }
            }
            Eigen::PartialPivLU<matrix_type> factorization(JtWJ_lambda);
            step = factorization.solve(JtWdy);

            if (options.use_geodesic())
            {
                p_new = parameters - epsilon * step;
                function_(p_new, y_p_new);

                y_tmp = J * step;

                tmp = Jt * (Dh * ((y_p_new - y_p) + epsilon * y_tmp));

                tmp = factorization.solve(tmp).eval();

                if (2. * l2_norm(tmp) < l2_norm(step) * alpha)
                {
                    step -= 0.5 * tmp;
                }
            }
            p_new = parameters - step;
            function_(p_new, y_p_new);
            auto x2_p_new = l2_norm(y_p_new);

            scalar_type alpha_quadratic = 0.;

            if (options.type() == levenberg_marquardt_solver_enum::QUADRATIC)
            {
                auto dot_product = step.dot(JtWdy);

                alpha_quadratic = dot_product / ((x2_p_new - x2_p) * 0.5 + 2.0 * dot_product);
                if (x2_p_new > x2_p)
                {
                    tmp = (-alpha_quadratic) * step;
                    tmp += parameters;

                    function_(tmp, y_tmp);
                    const auto norm = l2_norm(y_tmp);

                    if (x2_p > norm)
                    {
                        x2_p_new = norm;
                        p_new = tmp;
                        y_p_new = y_tmp;

                        if (accept_uphill_step)
                        {
                            step *= alpha_quadratic;
                        }
                    }
                }
            }

            const auto numerator_rho   = x2_p - x2_p_new;
            const auto denominator_rho = step.dot(lambda * diagonals.cwiseProduct(step) + JtWdy);

            bool update_step = ((numerator_rho > 0.) && (denominator_rho > 0.));

            if (accept_uphill_step && iteration > 0)
            {
                auto norm_previous_h = l2_norm(last_accepted_step);
                auto norm_h          = l2_norm(step);

                auto cos_theta =
                    !is_almost_zero(norm_h * norm_previous_h)
                        ? step.dot(last_accepted_step) / (norm_h * norm_previous_h)
                        : 0.;

                min_x2 = std::min(x2_p, min_x2);

                update_step = update_step || (1. - cos_theta) * x2_p_new < min_x2;
            }

            if (update_step)
            {
                parameters = p_new;
                y_p = y_p_new;
                auto previous_x2 = x2_p;
                x2_p             = x2_p_new;
                min_x2           = std::min(x2_p, min_x2);

                // Enhanced logging for accepted steps
                SOLVERS_LOG_IF(
                    INFO,
                    options.verbose(),
                    "LM Iter " << std::setw(3) << iteration << " | ACCEPTED STEP | "
                               << "f(x) = " << std::scientific << std::setprecision(3) << x2_p
                               << " | prev = " << std::scientific << std::setprecision(3)
                               << previous_x2 << " | improvement = " << std::scientific
                               << std::setprecision(2) << (previous_x2 - x2_p)
                               << " | lambda = " << std::scientific << std::setprecision(2)
                               << lambda << " | step_norm = " << std::scientific
                               << std::setprecision(2) << l2_norm(step));

                // decrease lambda == > Gauss - Newton method
                switch (options.type())
                {
                case levenberg_marquardt_solver_enum::LEVENBERG:
                    lambda = std::max(lambda / options.lambda_down_fac(), 1.e-7);
                    break;

                case levenberg_marquardt_solver_enum::QUADRATIC:
                    lambda = std::max(lambda / (1 + 2. * alpha_quadratic), 1.e-7);
                    break;

                case levenberg_marquardt_solver_enum::NIELSEN:
                    auto rho = numerator_rho / denominator_rho;
                    lambda *= std::fmax(1. / 3., 1. - std::fabs(pow(2. * rho - 1., 3.)));
                    nu = 2;
                    break;
                }

                jacobian(parameters, J);

                Jt    = J.transpose();
                JtWJ  = Jt * J;
                JtWdy = Jt * y_p;

                auto step_norm     = l2_norm(step);
                auto param_norm    = l2_norm(parameters);
                auto gradient_norm = l2_norm(JtWdy);

                parameters_converged =
                    parameters_converged || step_norm < param_norm * options.parameter_tolerance();

                gradient_converged =
                    gradient_converged || gradient_norm < options.gradient_tolerance();

                x2_converged = x2_converged || x2_p < options.function_tolerance();

                // Log convergence status if any criterion is met
                if (parameters_converged || gradient_converged || x2_converged)
                {
                    SOLVERS_LOG_IF(
                        INFO,
                        options.verbose(),
                        "LM Iter " << std::setw(3) << iteration << " | CONVERGENCE CHECK | "
                                   << "rel_step = " << std::scientific << std::setprecision(2)
                                   << (step_norm / param_norm)
                                   << " | grad_norm = " << std::scientific << std::setprecision(2)
                                   << gradient_norm << " | f(x) = " << std::scientific
                                   << std::setprecision(3) << x2_p);
                }

                stop = parameters_converged || gradient_converged || x2_converged;

                if (accept_uphill_step)
                {
                    last_accepted_step = step;
                }
            }
            else
            {
                // Enhanced logging for rejected steps
                SOLVERS_LOG_IF(
                    INFO,
                    options.verbose(),
                    "LM Iter " << std::setw(3) << iteration << " | REJECTED STEP | "
                               << "f(x) = " << std::scientific << std::setprecision(3) << x2_p
                               << " | trial = " << std::scientific << std::setprecision(3)
                               << x2_p_new << " | worsening = " << std::scientific
                               << std::setprecision(2) << (x2_p_new - x2_p)
                               << " | lambda = " << std::scientific << std::setprecision(2)
                               << lambda << " -> increasing");

                // increase lambda == > gradient descent method
                switch (options.type())
                {
                case levenberg_marquardt_solver_enum::LEVENBERG:
                    lambda = std::min(lambda * options.lambda_up_fac(), 1.e7);
                    break;

                case levenberg_marquardt_solver_enum::QUADRATIC:
                    lambda = lambda + std::fabs(0.5 * (x2_p - x2_p_new) / alpha_quadratic);
                    lambda = std::min(lambda, MAX_LAMBDA);
                    break;

                case levenberg_marquardt_solver_enum::NIELSEN:
                    lambda *= nu;
                    lambda = std::min(lambda, MAX_LAMBDA);
                    nu *= 2;
                    break;
                }

                // Log the new lambda value after adjustment
                SOLVERS_LOG_IF(
                    INFO,
                    options.verbose(),
                    "LM Iter " << std::setw(3) << iteration << " | lambda increased to "
                               << std::scientific << std::setprecision(2) << lambda);
                if (options.type() == levenberg_marquardt_solver_enum::NIELSEN &&
                    lambda >= MAX_LAMBDA)
                {
                    break;
                }
            }
        }
    }

    // Log final optimization status
    SOLVERS_LOG_IF(
        INFO,
        options.verbose(),
        "LM COMPLETED | iterations = " << iteration << " | final f(x) = " << std::scientific
                                       << std::setprecision(3) << x2_p << " | "
                                       << (x2_converged ? "FUNCTION_CONVERGED " : "")
                                       << (parameters_converged ? "PARAMETERS_CONVERGED " : "")
                                       << (gradient_converged ? "GRADIENT_CONVERGED " : "")
                                       << (iteration >= max_iter ? "MAX_ITERATIONS_REACHED " : ""));

    solver_output output(num_residuals_);

    output.update(x2_converged, parameters_converged, gradient_converged, iteration, y_p);



    return output;
}
}  // namespace solverslib
