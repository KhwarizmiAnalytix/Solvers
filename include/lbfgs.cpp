#include "lbfgs.h"

#include "solver_options/solver_options_bfgs.h"

namespace solverslib
{
namespace
{
template <typename T>
inline double l2_norm(T const& h)
{
    return h.norm();
}
template <typename T>
inline double l_max_norm(T const& h)
{
    return h.cwiseAbs().maxCoeff();
}
}  // namespace

template <lbfgs_line_search_type type>
class line_search
{
};

template <>
class line_search<lbfgs_line_search_type::NOCEDAL_WRIGHT>
{
    using scalar_type   = double;
    using size_type     = size_t;
    using vector_type   = Eigen::VectorXd;
    using function_type = std::function<scalar_type(vector_type const&, vector_type&)>;

public:
    static void search(  //NOLINT
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
                SOLVERS_CHECK(
                    step != step_hi,
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

                SOLVERS_CHECK(
                    step != step_lo,
                    "the line search routine failed, possibly due to insufficient numeric "
                    "preclaision");

                step_lo = step;
                fx_lo   = fx;
                dg_lo   = dg;
            }
        }
    }
};

template <>
class line_search<lbfgs_line_search_type::BACKTRACKING>
{
    using scalar_type   = double;
    using size_type     = size_t;
    using vector_type   = Eigen::VectorXd;
    using function_type = std::function<scalar_type(vector_type const&, vector_type&)>;

public:
    static void search(  //NOLINT
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

        SOLVERS_CHECK(
            dg_init < 0, "the moving direction increases the objective function value", dg_init);

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

            SOLVERS_CHECK(
                iter < param.max_iteration_linesearch(),
                "the line search routine reached the maximum number of iterations");

            SOLVERS_CHECK(
                step >= param.step_min() && step <= param.step_max(),
                "the line search step :",
                step,
                " is out of the boundaries. step_min_: ",
                param.step_min(),
                " step_max_ ",
                param.step_max());

            step *= width;
        }
    }
};

template <>
class line_search<lbfgs_line_search_type::BRACKETING>
{
    using scalar_type   = double;
    using size_type     = size_t;
    using vector_type   = Eigen::VectorXd;
    using function_type = std::function<scalar_type(vector_type const&, vector_type&)>;

public:
    static void search(  //NOLINT
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

            SOLVERS_CHECK(
                iter < param.max_iteration_linesearch(),
                "the line search routine reached the maximum number of iterations");

            SOLVERS_CHECK(
                step >= param.step_min(),
                "the line search step became smaller than the minimum value allowed");

            step = std::min(0.5 * (step_lo + step_hi), param.step_max());
        }
    }
};

LBFGS::LBFGS(
    int                  num_parameters,
    int                  num_residuals,
    LBFGS::function_type function,
    LBFGS::jacobian_type jacobian)
    : function_(std::move(function)),
      jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters),
      num_residuals_(num_residuals) {};

LBFGS::LBFGS(int num_parameters, int num_residuals, LBFGS::function_type function)
    : function_(std::move(function)),
      num_parameters_(num_parameters),
      num_residuals_(num_residuals) {};

optimization_algorithm_output LBFGS::solve(
    Eigen::VectorXd& parameters, const solver_options_bfgs& options) const
{
    SOLVERS_CHECK(num_parameters_ == parameters.size());

    auto jacobian = jacobian_;
    if (jacobian == nullptr)
    {
        auto bump = options.bump();

        jacobian = [this, bump](vector_type const& x, matrix_type& dy_dx)
        {
            auto number_of_targets    = num_residuals_;
            auto number_of_parameters = num_parameters_;

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

    vector_type y_p(num_residuals_);
    matrix_type J(num_residuals_, num_parameters_);

    auto lbfg_function = [this, &y_p, &J, &jacobian](vector_type const& x, vector_type& grad)
    {
        function_(x, y_p);
        double fx = l2_norm(y_p);
        fx *= fx;

        jacobian(x, J);
        grad = 2. * (J.transpose() * y_p);

        return fx;
    };

    auto dim = parameters.size();

    vector_type grad(dim);

    auto fx = lbfg_function(parameters, grad);

    auto x2_p                 = fx;
    auto x2_converged         = x2_p < options.function_tolerance();
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

    size_type iter     = 0;
    size_type iter_tau = 0;

    for (; !x2_converged && iter < options.max_num_iterations(); ++iter)
    {
        scalar_type step = 0.5;

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

        if (std::fabs(fx) < options.function_tolerance())
        {
            parameters = p_new;
            x2_converged = true;
            break;
        }

        if (l2_norm(parameters - p_new) <
            std::max(l2_norm(parameters), 1.) * options.parameter_tolerance())
        {
            parameters = p_new;
            parameters_converged = true;
            break;
        }

        if (l2_norm(grad) < options.gradient_tolerance())
        {
            parameters = p_new;
            gradient_converged = true;
            break;
        }

        v.row(iter_tau) = (p_new - parameters).transpose();
        r.row(iter_tau) = (grad - grad_old).transpose();

        q = grad;

        for (size_type j = 0; j <= iter_tau; ++j)
        {
            alpha[j] = (v.row(j).transpose()).dot(q) / (v.row(j).transpose()).dot(r.row(j).transpose());
            q        = q - alpha[j] * r.row(j).transpose();
        }
        const auto v_tau = v.row(iter_tau).transpose();
        const auto r_tau = r.row(iter_tau).transpose();

        q = q * (v_tau).dot(r_tau) / (r_tau).dot(r_tau);

        for (size_type j = 0; j <= iter_tau; ++j)
        {
            const auto offset = iter_tau - j;

            q = q +
                (alpha[offset] - (r.row(offset).transpose()).dot(q) / (v.row(offset).transpose()).dot(r.row(offset).transpose())) *
                    v.row(offset).transpose();
        }

        scalar_type sign = 1;
        sign             = std::copysign(sign, (q).dot(grad));

        direction = sign * q;

        parameters = p_new;
        grad_old = grad;

        ++iter_tau;
        if (iter_tau >= options.tau())
        {
            iter_tau = 0;
        }
    }

    optimization_algorithm_output output(num_residuals_);

    output.update(x2_converged, parameters_converged, gradient_converged, iter, y_p);

    return output;
}
}  // namespace solverslib
