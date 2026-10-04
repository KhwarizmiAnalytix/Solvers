#include "solvers/levenberg_marquardt_solver.h"

#include <iomanip>
#include <sstream>

#include "detail/support.h"
#include "solver_options/solver_options_lm.h"

namespace solverslib
{
namespace
{
// logging::strings::format only substitutes plain "{}" placeholders (no
// format specifiers), so the width/precision manipulators the diagnostic
// log lines below rely on are applied here, once, before handing the
// resulting string off to SOLVERS_LOG_IF.
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
}  // namespace

template <typename T> inline double l2_norm(T const& h)
{
    return h.norm();
}

levenberg_marquardt_solver::levenberg_marquardt_solver(size_t num_parameters,
    size_t                                                    num_residuals,
    levenberg_marquardt_solver::function_type                 function,
    levenberg_marquardt_solver::jacobian_type                 jacobian)
    : function_(std::move(function)), jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters), num_residuals_(num_residuals)
{
}

native_result levenberg_marquardt_solver::solve(
    vector_type& parameters, const solver_options_lm& options) const
{
    auto jacobian = jacobian_;
    if (jacobian == nullptr)
    {
        auto bump = options.finite_difference_step();

        jacobian = [this, bump](vector_type const& x, matrix_type& dy_dx)
        {
            auto number_of_parameters = x.size();

            SOLVERS_CHECK(dy_dx.cols() == number_of_parameters);

            auto number_of_targets = dy_dx.rows();

            vector_type y_plus  = make_vector(number_of_targets);
            vector_type y_minus = make_vector(number_of_targets);

            vector_type x_tmp = make_vector(number_of_parameters);
            x_tmp             = x;

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

    const auto max_iter                 = options.max_num_iterations();
    const auto bold_acceptance          = options.bold_acceptance();
    const auto bold_acceptance_exponent = options.bold_acceptance_exponent();
    const auto alpha                    = options.geodesic_acceleration_threshold();
    const auto epsilon                  = options.geodesic_acceleration_step();

    const auto Dh = 2. / (epsilon * epsilon);

    auto       lambda                              = options.initial_damping();
    auto       nu                                  = options.initial_rejection_multiplier();
    const auto damping_floor                       = options.damping_floor();
    const auto nielsen_damping_floor               = options.nielsen_damping_floor();
    const auto damping_ceiling                     = options.damping_ceiling();
    const auto levenberg_marquardt_damping_ceiling = options.levenberg_marquardt_damping_ceiling();
    const auto diagonal_scaling_floor              = options.diagonal_scaling_floor();
    const auto roundoff_noise_factor               = options.roundoff_noise_factor();

    vector_type y_p(m), y_p_new(m), y_tmp(m);
    vector_type JtWdy(n), p_new(n), last_accepted_velocity = vector_type::Zero(n), velocity(n);
    vector_type step(n), tmp(n), diagonals                 = vector_type::Ones(n);
    matrix_type J(m, n), Jt(n, m), JtWJ(n, n), JtWJ_lambda(n, n);

    function_(parameters, y_p);
    auto x2_p = l2_norm(y_p);
    // Smallest cost yet found, C(θ) = Σ r_m(θ)^2.
    auto min_cost             = y_p.squaredNorm();
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

        gradient_converged = JtWdy.norm() <= options.gradient_tolerance();
        stop               = gradient_converged;

        for (; !stop && iteration < max_iter; ++iteration)
        {
            SOLVERS_CHECK_FINITE_DEBUG(lambda);
            SOLVERS_CHECK_FINITE_DEBUG(nu);

            JtWJ_lambda = JtWJ;
            switch (options.type())
            {
            case levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = std::max(JtWJ(i, i), diagonal_scaling_floor);
                    JtWJ_lambda(i, i) += lambda * diagonals[i];
                }
                break;
            }
            case levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION:
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
            linear_system_solver factorization(JtWJ_lambda);
            step                = factorization.solve(JtWdy);
            velocity            = step;  // Stored with opposite sign; cosine is unchanged.
            bool geodesic_valid = step.allFinite();

            if (options.geodesic_acceleration())
            {
                p_new = parameters - epsilon * step;
                function_(p_new, y_p_new);

                y_tmp = J * step;

                // The subtraction in Eq. (19) loses precision near a solution.
                // Remove only remainders within a first-order roundoff bound;
                // otherwise noise divided by h^2 can reject every tiny step.
                vector_type remainder        = (y_p_new - y_p) + epsilon * y_tmp;
                vector_type coordinate_scale = J.cwiseAbs() * parameters.cwiseAbs();
                for (size_t i = 0; i < m; ++i)
                {
                    const double noise = roundoff_noise_factor *
                                         std::numeric_limits<double>::epsilon() *
                                         (std::abs(y_p_new[i]) + std::abs(y_p[i]) +
                                             std::abs(epsilon * y_tmp[i]) + coordinate_scale[i]);
                    if (std::abs(remainder[i]) <= noise)
                    {
                        remainder[i] = 0.;
                    }
                }
                tmp = Jt * (Dh * remainder);

                tmp = factorization.solve(tmp);

                // tmp = -a; Eq. (15) requires ||a|| <= alpha ||v||.
                // Reject the entire trial when the perturbation is too large.
                geodesic_valid =
                    geodesic_valid && tmp.allFinite() && l2_norm(tmp) <= l2_norm(velocity) * alpha;
                step += 0.5 * tmp;  // theta_new = theta + v + a/2.
            }
            p_new = parameters - step;
            function_(p_new, y_p_new);
            auto x2_p_new = l2_norm(y_p_new);

            scalar_type alpha_quadratic = 0.;

            if (options.type() == levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION)
            {
                auto dot_product = step.dot(JtWdy);

                alpha_quadratic = dot_product / ((y_p_new.squaredNorm() - y_p.squaredNorm()) * 0.5 +
                                                    2.0 * dot_product);
                if (geodesic_valid && x2_p_new > x2_p && std::isfinite(alpha_quadratic) &&
                    alpha_quadratic > 0.)
                {
                    tmp = (-alpha_quadratic) * step;
                    tmp += parameters;

                    function_(tmp, y_tmp);
                    const auto norm = l2_norm(y_tmp);

                    if (x2_p > norm)
                    {
                        x2_p_new = norm;
                        p_new    = tmp;
                        y_p_new  = y_tmp;

                        step *= alpha_quadratic;
                    }
                }
            }

            const auto numerator_rho = y_p.squaredNorm() - y_p_new.squaredNorm();
            // Twice the linearized cost reduction, evaluated at the actual
            // proposed step (including acceleration/interpolation).
            const auto denominator_rho = 2. * step.dot(JtWdy) - (J * step).squaredNorm();

            bool update_step = ((numerator_rho > 0.) && (denominator_rho > 0.));

            // Bold acceptance of an uphill step. Transtrum & Sethna,
            // "Improvements to the Levenberg-Marquardt algorithm for nonlinear
            // least-squares minimization", arXiv preprint (2012),
            // arXiv:1201.5885, Section 4. Downhill steps are accepted above.
            // An uphill step is accepted when the proposed velocity stays
            // aligned with the last accepted step:
            //   β_i = cos(v_new, v_old)                         (20)
            //   (1 - β_i)^b * C_{i+1} ≤ min(C_1, ..., C_i)        (22)
            // The paper's comparisons use b = 2.
            if (bold_acceptance && last_accepted_velocity.norm() > 0.)
            {
                const auto norm_old = l2_norm(last_accepted_velocity);
                const auto norm_new = l2_norm(velocity);

                auto beta = norm_new > 0. && norm_old > 0.
                                ? (velocity / norm_new).dot(last_accepted_velocity / norm_old)
                                : 0.;
                beta      = std::max(-1., std::min(1., beta));

                const auto cost_new = y_p_new.squaredNorm();
                update_step = update_step ||
                              std::pow(1. - beta, bold_acceptance_exponent) * cost_new <= min_cost;
            }

            update_step = update_step && geodesic_valid && step.allFinite() && p_new.allFinite() &&
                          y_p_new.allFinite() && std::isfinite(x2_p_new);

            if (update_step)
            {
                parameters       = p_new;
                y_p              = y_p_new;
                auto previous_x2 = x2_p;
                x2_p             = x2_p_new;
                min_cost         = std::min(y_p.squaredNorm(), min_cost);

                // Enhanced logging for accepted steps
                SOLVERS_LOG_IF(INFO,
                    options.verbose(),
                    "LM Iter {} | ACCEPTED STEP | f(x) = {} | prev = {} | improvement = {} | "
                    "lambda = {} | step_norm = {}",
                    fmt_iter(iteration),
                    fmt_sci(x2_p, 3),
                    fmt_sci(previous_x2, 3),
                    fmt_sci(previous_x2 - x2_p, 2),
                    fmt_sci(lambda, 2),
                    fmt_sci(l2_norm(step), 2));

                // decrease lambda == > Gauss - Newton method
                switch (options.type())
                {
                case levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT:
                    lambda = std::max(lambda / options.damping_decrease_factor(), damping_floor);
                    break;

                case levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION:
                    lambda = std::max(lambda / (1 + 2. * alpha_quadratic), damping_floor);
                    break;

                case levenberg_marquardt_solver_enum::NIELSEN:
                    // Bold acceptance can admit a trial without positive
                    // predicted reduction. Treat that as poor model agreement.
                    const auto rho = denominator_rho > 0. && std::isfinite(denominator_rho)
                                         ? numerator_rho / denominator_rho
                                         : 0.;
                    lambda *= std::fmax(1. / 3., 1. - pow(2. * rho - 1., 3.));
                    lambda = std::max(nielsen_damping_floor, std::min(lambda, damping_ceiling));
                    nu     = options.initial_rejection_multiplier();
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
                    parameters_converged ||
                    step_norm <= options.parameter_tolerance() *
                                     (param_norm + options.parameter_tolerance());

                gradient_converged =
                    gradient_converged || gradient_norm <= options.gradient_tolerance();

                x2_converged = x2_converged || x2_p < options.function_tolerance();

                // Log convergence status if any criterion is met
                if (parameters_converged || gradient_converged || x2_converged)
                {
                    SOLVERS_LOG_IF(INFO,
                        options.verbose(),
                        "LM Iter {} | CONVERGENCE CHECK | rel_step = {} | grad_norm = {} | f(x) = "
                        "{}",
                        fmt_iter(iteration),
                        fmt_sci(
                            step_norm / std::max(param_norm, std::numeric_limits<double>::min()),
                            2),
                        fmt_sci(gradient_norm, 2),
                        fmt_sci(x2_p, 3));
                }

                stop = parameters_converged || gradient_converged || x2_converged;

                if (bold_acceptance)
                {
                    last_accepted_velocity = velocity;
                }
            }
            else
            {
                // Enhanced logging for rejected steps
                SOLVERS_LOG_IF(INFO,
                    options.verbose(),
                    "LM Iter {} | REJECTED STEP | f(x) = {} | trial = {} | worsening = {} | "
                    "lambda = {} -> increasing",
                    fmt_iter(iteration),
                    fmt_sci(x2_p, 3),
                    fmt_sci(x2_p_new, 3),
                    fmt_sci(x2_p_new - x2_p, 2),
                    fmt_sci(lambda, 2));

                // increase lambda == > gradient descent method
                switch (options.type())
                {
                case levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT:
                    lambda = std::min(lambda * options.damping_increase_factor(),
                        levenberg_marquardt_damping_ceiling);
                    break;

                case levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION:
                    lambda = std::isfinite(alpha_quadratic) && alpha_quadratic > 0. &&
                                     std::isfinite(numerator_rho)
                                 ? lambda + std::fabs(0.5 * numerator_rho / alpha_quadratic)
                                 : lambda * options.damping_increase_factor();
                    lambda = std::min(lambda, damping_ceiling);
                    break;

                case levenberg_marquardt_solver_enum::NIELSEN:
                    lambda *= nu;
                    lambda = std::min(lambda, damping_ceiling);
                    nu *= 2;
                    break;
                }

                // Log the new lambda value after adjustment
                SOLVERS_LOG_IF(INFO,
                    options.verbose(),
                    "LM Iter {} | lambda increased to {}",
                    fmt_iter(iteration),
                    fmt_sci(lambda, 2));
                if (options.type() == levenberg_marquardt_solver_enum::NIELSEN &&
                    lambda >= damping_ceiling)
                {
                    break;
                }
            }
        }
    }

    // Log final optimization status
    SOLVERS_LOG_IF(INFO,
        options.verbose(),
        "LM COMPLETED | iterations = {} | final f(x) = {} | {}{}{}{}",
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

    return result;
}
}  // namespace solverslib
