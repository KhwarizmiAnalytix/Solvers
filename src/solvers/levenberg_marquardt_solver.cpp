#include "solvers/levenberg_marquardt_solver.h"

#include <iomanip>
#include <optional>
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
    function_type                                             function,
    jacobian_type                                             jacobian,
    double                                                    finite_difference_step,
    finite_difference_scale                                   difference_scale)
    : function_(std::move(function)), jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters), num_residuals_(num_residuals),
      fd_step_(finite_difference_step), fd_scale_(difference_scale)
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
            if (h <= 0.0)
            {
                SOLVERS_THROW("finite-difference step is not positive");
            }
            fd_parameters_[j] = x[j] + h;
            function_(fd_parameters_, fd_residual_);
            jacobian_matrix.col(static_cast<index_type>(j)) = (fd_residual_ - fd_base_) / h;
            fd_parameters_[j]                               = x[j];
        }
    };
}

native_result levenberg_marquardt_solver::solve(
    vector_type& parameters, const solver_options_lm& options) const
{
    std::string failure;  // non-empty once an evaluation fails fatally

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

    // Everything the iteration needs is allocated here, once per solve.
    const bool normal_equations =
        options.linear_solver() == levenberg_marquardt_linear_solver_enum::NORMAL_LDLT;
    const bool  geodesic = options.geodesic_acceleration();
    vector_type y_p(m), y_p_new(m), y_tmp(m);
    vector_type JtWdy(n), p_new(n), last_accepted_velocity = vector_type::Zero(n), velocity(n);
    vector_type step(n), tmp(n), diagonals                 = vector_type::Ones(n);
    vector_type damping(n), jtv(n), abs_x(geodesic ? n : 0);
    vector_type remainder(geodesic ? m : 0), coordinate_scale(geodesic ? m : 0), j_step(m);
    matrix_type J(m, n), JtWJ = matrix_type::Zero(n, n);
    matrix_type abs_J(geodesic ? m : 0, geodesic ? n : 0);
    damped_step_solver steps(static_cast<index_type>(m),
        static_cast<index_type>(n),
        normal_equations ? damped_step_method::normal_ldlt : damped_step_method::augmented_qr);

    // J^T J (full product for the normal equations; only its diagonal, which the
    // damping scaling reads, for the QR form) and J^T y_p.
    const auto refresh_normal_terms = [&]()
    {
        if (normal_equations)
        {
            JtWJ.noalias() = J.transpose() * J;
        }
        else
        {
            JtWJ.diagonal() = J.colwise().squaredNorm();
        }
        JtWdy.noalias() = J.transpose() * y_p;
    };
    const double function_tolerance = options.function_tolerance();
    // F = 0.5 * ||r||^2, the objective the API documents.
    const auto function_converged = [function_tolerance](double residual_norm)
    { return 0.5 * residual_norm * residual_norm < function_tolerance; };

    function_(parameters, y_p);
    if (!y_p.allFinite())
    {
        SOLVERS_THROW("non-finite residuals at the initial point");
    }
    auto x2_p = l2_norm(y_p);
    // Smallest cost yet found, C(θ) = Σ r_m(θ)^2.
    auto min_cost             = y_p.squaredNorm();
    auto x2_converged         = function_converged(x2_p);
    bool gradient_converged   = false;
    bool parameters_converged = false;

    size_t iteration = 0;

    size_t                accepted_steps = 0;
    size_t                rejected_steps = 0;
    bool                  stalled        = false;
    std::optional<double> last_step_norm;
    std::optional<double> final_gradient_norm;

    if (!x2_converged)
    {
        jacobian_(parameters, J);
        if (!J.allFinite())
        {
            SOLVERS_THROW("non-finite Jacobian at the initial point");
        }

        refresh_normal_terms();

        final_gradient_norm = JtWdy.norm();
        gradient_converged  = JtWdy.norm() <= options.gradient_tolerance();
        bool stop           = gradient_converged;

        for (; !stop && iteration < max_iter; ++iteration)
        {
            SOLVERS_CHECK_FINITE_DEBUG(lambda);
            SOLVERS_CHECK_FINITE_DEBUG(nu);

            switch (options.type())
            {
            case levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = std::max(JtWJ(i, i), diagonal_scaling_floor);
                    damping[i]   = lambda * diagonals[i];
                }
                break;
            }
            case levenberg_marquardt_solver_enum::QUADRATIC_INTERPOLATION:
            {
                damping.setConstant(lambda);
                break;
            }
            case levenberg_marquardt_solver_enum::NIELSEN:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = std::max(JtWJ(i, i), diagonals[i]);
                    damping[i]   = lambda * diagonals[i];
                }
                break;
            }
            }
            // A breakdown of the factorization is cured by more damping, so it
            // is handled as a rejected step rather than a failure.
            const bool factored = steps.factor(J, JtWJ, damping);
            if (factored)
            {
                steps.solve(y_p, JtWdy, step);
            }
            else
            {
                step.setZero();
            }
            velocity            = step;  // Stored with opposite sign; cosine is unchanged.
            bool geodesic_valid = factored && step.allFinite();

            if (factored && geodesic)
            {
                p_new = parameters - epsilon * step;
                function_(p_new, y_p_new);
                if (!y_p_new.allFinite())
                {
                    geodesic_valid = false;
                }

                y_tmp.noalias() = J * step;

                // The subtraction in Eq. (19) loses precision near a solution.
                // Remove only remainders within a first-order roundoff bound;
                // otherwise noise divided by h^2 can reject every tiny step.
                remainder                  = (y_p_new - y_p) + epsilon * y_tmp;
                abs_J                      = J.cwiseAbs();
                abs_x                      = parameters.cwiseAbs();
                coordinate_scale.noalias() = abs_J * abs_x;
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
                remainder *= Dh;
                jtv.noalias() = J.transpose() * remainder;
                steps.solve(remainder, jtv, tmp);

                // tmp = -a; Eq. (15) requires ||a|| <= alpha ||v||.
                // Reject the entire trial when the perturbation is too large.
                geodesic_valid =
                    geodesic_valid && tmp.allFinite() && l2_norm(tmp) <= l2_norm(velocity) * alpha;
                step += 0.5 * tmp;  // theta_new = theta + v + a/2.
            }
            p_new            = parameters - step;
            bool trial_valid = false;
            if (!factored)
            {
                y_p_new = y_p;  // nothing to evaluate; the step is rejected below
            }
            else
            {
                function_(p_new, y_p_new);
                trial_valid = y_p_new.allFinite();
            }
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
                    const bool interpolation_valid = y_tmp.allFinite();
                    const auto norm                = l2_norm(y_tmp);

                    if (interpolation_valid && x2_p > norm)
                    {
                        trial_valid = true;
                        x2_p_new    = norm;
                        p_new       = tmp;
                        y_p_new     = y_tmp;

                        step *= alpha_quadratic;
                    }
                }
            }

            const auto numerator_rho = y_p.squaredNorm() - y_p_new.squaredNorm();
            // Twice the linearized cost reduction, evaluated at the actual
            // proposed step (including acceleration/interpolation).
            j_step.noalias()           = J * step;
            const auto denominator_rho = 2. * step.dot(JtWdy) - j_step.squaredNorm();

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
                update_step         = update_step ||
                              std::pow(1. - beta, bold_acceptance_exponent) * cost_new <= min_cost;
            }

            update_step = update_step && trial_valid && geodesic_valid && step.allFinite() &&
                          p_new.allFinite() && y_p_new.allFinite() && std::isfinite(x2_p_new);

            if (update_step)
            {
                ++accepted_steps;
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

                jacobian_(parameters, J);
                SOLVERS_CHECK(J.allFinite(), "non-finite Jacobian at an accepted point");

                refresh_normal_terms();

                auto step_norm      = l2_norm(step);
                auto param_norm     = l2_norm(parameters);
                auto gradient_norm  = l2_norm(JtWdy);
                last_step_norm      = step_norm;
                final_gradient_norm = gradient_norm;

                parameters_converged =
                    parameters_converged ||
                    step_norm <= options.parameter_tolerance() *
                                     (param_norm + options.parameter_tolerance());

                gradient_converged =
                    gradient_converged || gradient_norm <= options.gradient_tolerance();

                x2_converged = x2_converged || function_converged(x2_p);

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
                ++rejected_steps;
                // A rejection while the damping is already at its ceiling cannot
                // be cured by more damping: the iterate is stuck.
                const auto active_ceiling =
                    options.type() == levenberg_marquardt_solver_enum::LEVENBERG_MARQUARDT
                        ? levenberg_marquardt_damping_ceiling
                        : damping_ceiling;
                const bool was_at_ceiling = lambda >= active_ceiling;
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
                if (was_at_ceiling && !factored)
                {
                    failure = "linear solve failed even at the damping ceiling";
                    break;
                }
                if (was_at_ceiling || (options.type() == levenberg_marquardt_solver_enum::NIELSEN &&
                                          lambda >= damping_ceiling))
                {
                    stalled = true;
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
        result.message = "damping reached its ceiling without finding an acceptable step";
    }
    result.accepted_steps = accepted_steps;
    result.rejected_steps = rejected_steps;
    result.step_norm      = last_step_norm;
    result.gradient_norm  = final_gradient_norm;

    return result;
}
}  // namespace solverslib
