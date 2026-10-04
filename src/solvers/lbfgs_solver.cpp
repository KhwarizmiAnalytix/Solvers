#include "solvers/lbfgs_solver.h"

#include <optional>
#include <string>

#include "detail/native_evaluation.h"
#include "solver_options/solver_options_bfgs.h"

namespace solverslib
{
namespace
{
template <typename T> inline double l2_norm(T const& h)
{
    return h.norm();
}

// Thrown from inside the line-search callback to unwind to solve() when an
// evaluation fails fatally; never escapes the kernel.
struct evaluation_failure
{
    std::string message;
};

}  // namespace

template <lbfgs_line_search_type type> class line_search
{
};

template <> class line_search<lbfgs_line_search_type::NOCEDAL_WRIGHT>
{
    using scalar_type   = double;
    using size_type     = size_t;
    using function_type = std::function<scalar_type(vector_type const&, vector_type&)>;

public:
    static void search(  // NOLINT
        const function_type&       f,
        scalar_type&               fx,
        vector_type&               x,
        vector_type&               grad,
        scalar_type&               step,
        const vector_type&         direction,
        const vector_type&         xp,
        const solver_options_bfgs& param)
    {
        const auto expansion = static_cast<scalar_type>(5.);
        const auto fx_init   = fx;
        const auto dg_init   = -grad.dot(direction);
        SOLVERS_CHECK(dg_init <= 0, "the moving direction increases the objective function value");

        const auto dg_test  = param.linesearch_tolerance();
        const auto dg_wolfe = -param.linesearch_wolfe() * dg_init;

        scalar_type step_hi = param.step_max();
        x                   = xp - step_hi * direction;
        scalar_type fx_hi   = f(x, grad);

        scalar_type step_lo = param.step_min();
        scalar_type fx_lo   = fx_init;

        scalar_type dg_lo = dg_init;

        size_type iter = 0;
        for (; iter < param.max_iteration_linesearch(); ++iter)
        {
            x  = xp - step * direction;
            fx = f(x, grad);

            const scalar_type dg = -grad.dot(direction);

            if (fx - fx_init > step * dg_test || (0 < step_lo && fx >= fx_lo))
            {
                step_hi = step;
                fx_hi   = fx;
                break;
            }

            if (std::abs(dg) <= dg_wolfe)
            {
                return;
            }

            step_hi = step_lo;
            fx_hi   = fx_lo;
            step_lo = step;
            fx_lo   = fx;
            dg_lo   = dg;

            if (dg >= 0)
            {
                break;
            }

            step *= expansion;
        }

        if (step_hi < step_lo)
        {
            std::swap(fx_hi, fx_lo);
            std::swap(step_hi, step_lo);
        }

        for (; iter < param.max_iteration_linesearch(); ++iter)
        {
            step =
                (fx_hi - fx_lo) * step_lo - 0.5 * (step_hi * step_hi - step_lo * step_lo) * dg_lo;
            step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo;

            if (step <= step_lo || step >= step_hi)
            {
                step = 0.5 * (step_lo + step_hi);
            }

            x  = xp - step * direction;
            fx = f(x, grad);

            const scalar_type dg = -grad.dot(direction);

            if (fx - fx_init > step * dg_test || fx >= fx_lo)
            {
                SOLVERS_CHECK(step != step_hi,
                    "the line search routine failed, possibly due to insufficient numeric "
                    "precision");

                step_hi = step;
                fx_hi   = fx;
            }
            else
            {
                if (std::abs(dg) <= dg_wolfe)
                {
                    return;
                }

                if (dg * (step_hi - step_lo) >= 0)
                {
                    step_hi = step_lo;
                    fx_hi   = fx_lo;
                }

                SOLVERS_CHECK(step != step_lo,
                    "the line search routine failed, possibly due to insufficient numeric "
                    "preclaision");

                step_lo = step;
                fx_lo   = fx;
                dg_lo   = dg;
            }
        }
    }
};

template <> class line_search<lbfgs_line_search_type::BACKTRACKING>
{
    using scalar_type   = double;
    using size_type     = size_t;
    using function_type = std::function<scalar_type(vector_type const&, vector_type&)>;

public:
    static void search(  // NOLINT
        const function_type&       f,
        scalar_type&               fx,
        vector_type&               x,
        vector_type&               grad,
        scalar_type&               step,
        const vector_type&         direction,
        const vector_type&         xp,
        const solver_options_bfgs& param)
    {
        const scalar_type dec = 0.5;
        const scalar_type inc = 2.1;

        SOLVERS_CHECK(step > scalar_type(0), "'step' must be positive");

        const scalar_type fx_init = fx;
        const scalar_type dg_init = -grad.dot(direction);

        SOLVERS_CHECK(dg_init < 0,
            "the moving direction increases the objective function value: {}",
            dg_init);

        const scalar_type dg_test = param.linesearch_tolerance() * dg_init;
        scalar_type       width;

        for (size_type iter = 0; iter < param.max_iteration_linesearch(); ++iter)
        {
            x  = xp - step * direction;
            fx = f(x, grad);

            if (fx > fx_init + step * dg_test)
            {
                width = dec;
            }
            else
            {
                // Armijo condition is met
                if (param.method_type() == lbfgs_line_search_method_type::ARMIJO)
                {
                    break;
                }

                const scalar_type dg = -grad.dot(direction);
                if (dg < param.linesearch_wolfe() * dg_init)
                {
                    width = inc;
                }
                else
                {
                    // Regular Wolfe condition is met
                    if (param.method_type() == lbfgs_line_search_method_type::WOLFE)
                    {
                        break;
                    }

                    if (dg > -param.linesearch_wolfe() * dg_init)
                    {
                        width = dec;
                    }
                    else
                    {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            SOLVERS_CHECK(iter < param.max_iteration_linesearch(),
                "the line search routine reached the maximum number of iterations");

            SOLVERS_CHECK(step >= param.step_min() && step <= param.step_max(),
                "the line search step: {} is out of the boundaries. step_min_: {} step_max_: {}",
                step,
                param.step_min(),
                param.step_max());

            step *= width;
        }
    }
};

template <> class line_search<lbfgs_line_search_type::BRACKETING>
{
    using scalar_type   = double;
    using size_type     = size_t;
    using function_type = std::function<scalar_type(vector_type const&, vector_type&)>;

public:
    static void search(  // NOLINT
        const function_type&       f,
        scalar_type&               fx,
        vector_type&               x,
        vector_type&               grad,
        scalar_type&               step,
        const vector_type&         direction,
        const vector_type&         xp,
        const solver_options_bfgs& param)
    {
        const scalar_type fx_init = fx;
        const scalar_type dg_init = -grad.dot(direction);

        SOLVERS_CHECK(dg_init <= 0, "the moving direction increases the objective function value");

        const scalar_type dg_test = param.linesearch_tolerance() * dg_init;

        scalar_type step_lo = param.step_min();
        scalar_type step_hi = param.step_max();

        for (size_type iter = 0; iter < param.max_iteration_linesearch(); ++iter)
        {
            x  = xp - step * direction;
            fx = f(x, grad);

            if (fx > fx_init + step * dg_test)
            {
                step_hi = step;
            }
            else
            {
                if (param.method_type() == lbfgs_line_search_method_type::ARMIJO)
                {
                    break;
                }

                const scalar_type dg = -grad.dot(direction);
                if (dg < param.linesearch_wolfe() * dg_init)
                {
                    step_lo = step;
                }
                else
                {
                    if (param.method_type() == lbfgs_line_search_method_type::WOLFE)
                    {
                        break;
                    }

                    if (dg > -param.linesearch_wolfe() * dg_init)
                    {
                        step_hi = step;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            SOLVERS_CHECK(step_lo < step_hi, "step min is bigger than step max");

            SOLVERS_CHECK(iter < param.max_iteration_linesearch(),
                "the line search routine reached the maximum number of iterations");

            SOLVERS_CHECK(step >= param.step_min(),
                "the line search step became smaller than the minimum value allowed");

            step = std::min(0.5 * (step_lo + step_hi), param.step_max());
        }
    }
};

lbfgs_solver::lbfgs_solver(size_type num_parameters,
    size_type                        num_residuals,
    lbfgs_solver::function_type      function,
    lbfgs_solver::jacobian_type      jacobian)
    : function_(std::move(function)), jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters), num_residuals_(num_residuals) {};

lbfgs_solver::lbfgs_solver(
    size_type num_parameters, objective_type objective, gradient_type gradient)
    : num_parameters_(num_parameters), num_residuals_(0), objective_(std::move(objective)),
      gradient_(std::move(gradient)), scalar_mode_(true) {};

lbfgs_solver::lbfgs_solver(api::detail::residual_evaluator& evaluator)
    : num_parameters_(evaluator.metadata().num_parameters),
      num_residuals_(evaluator.metadata().num_residuals), evaluator_(&evaluator)
{
}

lbfgs_solver::lbfgs_solver(api::detail::gradient_evaluator& evaluator)
    : num_parameters_(evaluator.num_parameters()), num_residuals_(0), scalar_mode_(true),
      gradient_evaluator_(&evaluator)
{
}

native_result lbfgs_solver::solve(vector_type& parameters, const solver_options_bfgs& options) const
{
    try
    {
        return run(parameters, options);
    }
    catch (const evaluation_failure& failure)
    {
        native_result result;
        result.status        = native_convergence::numerical_failure;
        result.residual_norm = std::numeric_limits<double>::quiet_NaN();
        result.message       = failure.message;
        return result;
    }
}

native_result lbfgs_solver::run(vector_type& parameters, const solver_options_bfgs& options) const
{
    SOLVERS_CHECK(num_parameters_ == parameters.size());
    using api::detail::evaluation_status;

    // Scalar-objective mode: the caller supplied f(x) and ∇f(x) directly,
    // so the L-BFGS loop operates on the true objective without a residual
    // wrapping layer. Inspired by PyTorch's LBFGS optimizer, which always
    // works with a scalar closure.
    std::function<scalar_type(vector_type const&, vector_type&)> lbfg_function;

    // These are only used in residual mode but must live until solve returns.
    vector_type y_p;
    matrix_type J;

    // Wrappers built from raw callbacks live for this call only.
    std::unique_ptr<api::detail::gradient_evaluator> owned_gradient;
    native_evaluator_binding                         binding;

    if (scalar_mode_)
    {
        api::detail::gradient_evaluator* evaluator = gradient_evaluator_;
        if (evaluator == nullptr)
        {
            owned_gradient = std::make_unique<api::detail::callback_gradient_evaluator>(
                num_parameters_, objective_, gradient_, api::derivative_mode::supplied);
            evaluator = owned_gradient.get();
        }
        lbfg_function = [evaluator](vector_type const& x, vector_type& grad)
        {
            double     fx     = 0.0;
            const auto status = evaluator->evaluate(x, fx, &grad);
            if (status == evaluation_status::fatal_error)
            {
                throw evaluation_failure{"objective/gradient evaluation: " +
                                         evaluator->last_error().value_or("evaluation failed")};
            }
            // invalid_trial passes the (non-finite) value on so the line search
            // shortens its step, as it always has.
            return fx;
        };
    }
    else
    {
        binding = bind_native_evaluator(
            evaluator_, num_parameters_, num_residuals_, function_, jacobian_, options.bump());
        api::detail::residual_evaluator* evaluator = binding.evaluator;

        y_p.resize(num_residuals_);
        J.resize(num_residuals_, num_parameters_);

        lbfg_function = [evaluator, &y_p, &J](vector_type const& x, vector_type& grad)
        {
            auto status = evaluator->evaluate(x, y_p);
            if (status == evaluation_status::fatal_error)
            {
                throw evaluation_failure{
                    evaluation_failure_message(*evaluator, status, "residual evaluation")};
            }
            double fx = l2_norm(y_p);
            fx *= fx;

            status = evaluator->jacobian(x, y_p, J);
            if (status == evaluation_status::fatal_error)
            {
                throw evaluation_failure{
                    evaluation_failure_message(*evaluator, status, "Jacobian evaluation")};
            }
            grad = 2. * (J.transpose() * y_p);

            return fx;
        };
    }

    auto dim = parameters.size();

    vector_type grad(dim);

    auto fx = lbfg_function(parameters, grad);

    auto x2_p = fx;
    // A scalar objective may be shifted by an arbitrary constant.  Its
    // absolute value is therefore not a valid convergence test; use step and
    // gradient criteria below.  Residual mode retains the historical norm test.
    auto x2_converged         = !scalar_mode_ && 0.5 * x2_p < options.function_tolerance();
    bool gradient_converged   = false;
    bool parameters_converged = false;

    vector_type p_new(dim);
    vector_type q(dim);
    vector_type direction(dim);
    direction = grad;

    vector_type grad_old(dim);
    grad_old = grad;

    matrix_type v(options.tau(), dim);
    matrix_type r(options.tau(), dim);
    vector_type alpha(options.tau());

    size_type             iter           = 0;
    size_type             iter_tau       = 0;
    std::size_t           accepted_steps = 0;
    bool                  stalled        = false;
    std::string           stall_message;
    std::optional<double> last_step_norm;

    for (; !x2_converged && iter < options.max_num_iterations(); ++iter)
    {
        const scalar_type previous_fx = fx;
        scalar_type       step        = 0.5;

        try
        {
            switch (options.type())
            {
            case lbfgs_line_search_type::NOCEDAL_WRIGHT:
                line_search<lbfgs_line_search_type::NOCEDAL_WRIGHT>::search(
                    lbfg_function, fx, p_new, grad, step, direction, parameters, options);
                break;
            case lbfgs_line_search_type::BACKTRACKING:
                line_search<lbfgs_line_search_type::BACKTRACKING>::search(
                    lbfg_function, fx, p_new, grad, step, direction, parameters, options);
                break;
            case lbfgs_line_search_type::BRACKETING:
                line_search<lbfgs_line_search_type::BRACKETING>::search(
                    lbfg_function, fx, p_new, grad, step, direction, parameters, options);
                break;
            }
        }
        catch (const evaluation_failure&)
        {
            throw;  // reported by solve() as a numerical failure
        }
        catch (const std::exception& e)
        {
            // The line search gave up (no step satisfies its conditions); the
            // last accepted iterate is still valid.
            stalled       = true;
            stall_message = e.what();
            break;
        }
        ++accepted_steps;
        last_step_norm = l2_norm(p_new - parameters);

        if (!scalar_mode_ && 0.5 * std::fabs(fx) < options.function_tolerance())
        {
            parameters   = p_new;
            x2_converged = true;
            break;
        }

        // For scalar objectives use improvement, rather than the absolute
        // objective value, so adding a constant cannot change convergence.
        if (scalar_mode_ &&
            std::fabs(fx - previous_fx) <=
                options.function_tolerance() * std::max(std::fabs(previous_fx), scalar_type(1.0)))
        {
            parameters   = p_new;
            x2_converged = true;
            break;
        }

        if (l2_norm(parameters - p_new) <
            std::max(l2_norm(parameters), 1.) * options.parameter_tolerance())
        {
            parameters           = p_new;
            parameters_converged = true;
            break;
        }

        if (l2_norm(grad) < options.gradient_tolerance())
        {
            parameters         = p_new;
            gradient_converged = true;
            break;
        }

        v.row(iter_tau) = (p_new - parameters).transpose();
        r.row(iter_tau) = (grad - grad_old).transpose();

        q = grad;

        for (size_type j = 0; j <= iter_tau; ++j)
        {
            alpha[j] =
                (v.row(j).transpose()).dot(q) / (v.row(j).transpose()).dot(r.row(j).transpose());
            q = q - alpha[j] * r.row(j).transpose();
        }
        const auto v_tau = v.row(iter_tau).transpose();
        const auto r_tau = r.row(iter_tau).transpose();

        q = q * (v_tau).dot(r_tau) / (r_tau).dot(r_tau);

        for (size_type j = 0; j <= iter_tau; ++j)
        {
            const auto offset = iter_tau - j;

            q = q +
                (alpha[offset] - (r.row(offset).transpose()).dot(q) /
                                     (v.row(offset).transpose()).dot(r.row(offset).transpose())) *
                    v.row(offset).transpose();
        }

        scalar_type sign = 1;
        sign             = std::copysign(sign, (q).dot(grad));

        direction = sign * q;

        parameters = p_new;
        grad_old   = grad;

        ++iter_tau;
        if (iter_tau >= options.tau())
        {
            iter_tau = 0;
        }
    }

    if (stalled)
    {
        // The failed search left trial values in fx/grad/y_p; restore them to
        // the last accepted point.
        fx = lbfg_function(parameters, grad);
    }

    native_result result;
    result.iterations     = iter;
    result.residual_norm  = scalar_mode_ ? std::sqrt(std::fabs(fx)) : l2_norm(y_p);
    result.accepted_steps = accepted_steps;
    result.step_norm      = last_step_norm;
    result.gradient_norm  = l2_norm(grad);
    if (stalled)
    {
        result.status  = native_convergence::stalled;
        result.message = "line search failed: " + stall_message;
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

    return result;
}
}  // namespace solverslib
