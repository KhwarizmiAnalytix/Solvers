#include "levenberg_marquardt.h"

#include "common/constants.h"
#include "optimization_algorithm/solver_options/solver_options_lm.h"

namespace quarisma
{
namespace
{
template <typename T>
inline double l2_norm(T const& h)
{
    return sqrt(accumulate(sqr(h)));
}
}  // namespace

levenberg_marquardt::levenberg_marquardt(
    int                                num_parameters,
    int                                num_residuals,
    levenberg_marquardt::function_type function,
    levenberg_marquardt::jacobian_type jacobian)
    : function_(std::move(function)),
      jacobian_(std::move(jacobian)),
      num_parameters_(num_parameters),
      num_residuals_(num_residuals)
{
}

levenberg_marquardt::levenberg_marquardt(
    int num_parameters, int num_residuals, levenberg_marquardt::function_type function)
    : function_(std::move(function)), num_parameters_(num_parameters), num_residuals_(num_residuals)
{
}

optimization_algorithm_output levenberg_marquardt::solve(
    vector<double>& parameters, const solver_options_lm& options) const
{
    QUARISMA_CHECK(num_parameters_ == parameters.size());

    auto jacobian = jacobian_;

    if (jacobian == nullptr)
    {
        auto bump = options.bump();

        jacobian = [this, bump](vector_type const& x, matrix_type& dy_dx)
        {
            auto number_of_parameters = x.size();

            QUARISMA_CHECK(dy_dx.columns() == number_of_parameters);

            auto number_of_targets = dy_dx.rows();

            vector_type y_plus(number_of_targets);
            vector_type y_minus(number_of_targets);

            vector_type x_tmp(number_of_parameters);
            x_tmp.deepcopy(x);

            for (size_t i = 0; i < number_of_parameters; ++i)
            {
                x_tmp[i] += bump;

                function_(x_tmp, y_plus);

                x_tmp[i] -= 2 * bump;
                function_(x_tmp, y_minus);

                for (size_t j = 0; j < y_plus.size(); ++j)
                {
                    dy_dx[j][i] = 0.5 * (y_plus[j] - y_minus[j]) / bump;
                }

                x_tmp[i] = x[i];
            }
        };
    }

    const auto n = num_parameters_;
    const auto m = num_residuals_;

    const auto max_iter           = options.max_num_iterations();
    const auto accept_uphill_step = options.accept_uphill_step();
    const auto alpha              = options.alpha();
    const auto epsilon            = options.epsilon();

    const auto Dh = 2. / (epsilon * epsilon);

    auto lambda = options.lambda();
    auto nu     = options.nu();

    std::vector<quarisma_int> pivot(n + 1);

    const auto total_size = 6 * n + 3 * m + 2 * m * n + 2 * n * n;

    auto* data = allocator<scalar_type>::allocate(total_size, device_type::CPU);

    memset(data, 0, total_size * sizeof(scalar_type));

    auto* iter = data;

    vector_type y_p(iter, m);
    iter += m;

    vector_type y_p_new(iter, m);
    iter += m;

    vector_type y_tmp(iter, m);
    iter += m;

    vector_type JtWdy(iter, n);
    iter += n;

    vector_type p_new(iter, n);
    iter += n;

    vector_type last_accepted_step(iter, n);
    iter += n;

    vector_type step(iter, n);
    iter += n;

    vector_type tmp(iter, n);
    iter += n;

    vector_type diagonals(iter, n);
    iter += n;

    matrix_type J(iter, m, n);
    iter += static_cast<ptrdiff_t>(m * n);

    matrix_type Jt(iter, n, m);
    iter += static_cast<ptrdiff_t>(m * n);

    matrix_type JtWJ(iter, n, n);
    iter += static_cast<ptrdiff_t>(n * n);

    matrix_type JtWJ_lambda(iter, n, n);

    diagonals = 1.;

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

        Jt    = transpose(J);
        JtWJ  = Jt * J;
        JtWdy = Jt * y_p;

        auto min_x2 = x2_p;

        for (; !stop && iteration < max_iter; ++iteration)
        {
            JtWJ_lambda.deepcopy(JtWJ);
            switch (options.type())
            {
            case levenberg_marquardt_type::LEVENBERG:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = JtWJ[i][i];
                    JtWJ_lambda[i][i] *= (1 + lambda);
                }
                break;
            }
            case levenberg_marquardt_type::QUADRATIC:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    JtWJ_lambda[i][i] += lambda;
                }
                break;
            }
            case levenberg_marquardt_type::NIELSEN:
            {
                for (size_t i = 0; i < n; ++i)
                {
                    diagonals[i] = std::max(JtWJ[i][i], diagonals[i]);
                    JtWJ_lambda[i][i] += lambda * diagonals[i];
                }
                break;
            }
            }
            step.deepcopy(JtWdy);
            linear_solver(
                JtWJ_lambda.begin(),
                pivot.data(),
                static_cast<quarisma_int>(n),
                step.data(),
                linear_solver_type::LU_LINEAR_SOLVER);

            if (options.use_geodesic())
            {
                p_new = parameters - epsilon * step;
                function_(p_new, y_p_new);

                y_tmp = J * step;

                tmp = Jt * (Dh * ((y_p_new - y_p) + epsilon * y_tmp));

                linear_solver(
                    JtWJ_lambda.begin(),
                    pivot.data(),
                    static_cast<quarisma_int>(n),
                    tmp.data(),
                    linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);

                if (2. * l2_norm(tmp) < l2_norm(step) * alpha)
                {
                    step -= 0.5 * tmp;
                }
            }
            p_new = parameters - step;
            function_(p_new, y_p_new);
            auto x2_p_new = l2_norm(y_p_new);

            scalar_type alpha_quadratic = 0.;

            if (options.type() == levenberg_marquardt_type::QUADRATIC)
            {
                auto dot_product = accumulate(step * JtWdy);

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
                        p_new.deepcopy(tmp);
                        y_p_new.deepcopy(y_tmp);

                        if (accept_uphill_step)
                        {
                            step *= alpha_quadratic;
                        }
                    }
                }
            }

            const auto numerator_rho   = x2_p - x2_p_new;
            const auto denominator_rho = accumulate(step * (lambda * diagonals * step + JtWdy));

            bool update_step = ((numerator_rho > 0.) && (denominator_rho > 0.));

            if (accept_uphill_step && iteration > 0)
            {
                auto norm_previous_h = l2_norm(last_accepted_step);
                auto norm_h          = l2_norm(step);

                auto cos_theta =
                    !is_almost_zero(norm_h * norm_previous_h)
                        ? accumulate(step * last_accepted_step) / (norm_h * norm_previous_h)
                        : 0.;

                min_x2 = std::min(x2_p, min_x2);

                update_step = update_step || (1. - cos_theta) * x2_p_new < min_x2;
            }

            if (update_step)
            {
                parameters.deepcopy(p_new);
                y_p.deepcopy(y_p_new);
                x2_p   = x2_p_new;
                min_x2 = std::min(x2_p, min_x2);

                // decrease lambda == > Gauss - Newton method
                switch (options.type())
                {
                case levenberg_marquardt_type::LEVENBERG:
                    lambda = std::max(lambda / options.lambda_down_fac(), 1.e-7);
                    break;

                case levenberg_marquardt_type::QUADRATIC:
                    lambda = std::max(lambda / (1 + 2. * alpha_quadratic), 1.e-7);
                    break;

                case levenberg_marquardt_type::NIELSEN:
                    auto rho = numerator_rho / denominator_rho;
                    lambda *= std::fmax(1. / 3., 1. - std::fabs(pow(2. * rho - 1., 3.)));
                    nu = 2;
                    break;
                }

                jacobian(parameters, J);

                Jt    = transpose(J);
                JtWJ  = Jt * J;
                JtWdy = Jt * y_p;

                parameters_converged =
                    parameters_converged ||
                    l2_norm(step) < l2_norm(parameters) * options.parameter_tolerance();

                gradient_converged =
                    gradient_converged || l2_norm(JtWdy) < options.gradient_tolerance();

                x2_converged = x2_converged || x2_p < options.function_tolerance();

                stop = parameters_converged || gradient_converged || x2_converged;

                if (accept_uphill_step)
                {
                    last_accepted_step.deepcopy(step);
                }
            }
            else
            {
                // increase lambda == > gradient descent method
                switch (options.type())
                {
                case levenberg_marquardt_type::LEVENBERG:
                    lambda = std::min(lambda * options.lambda_up_fac(), 1.e7);
                    break;

                case levenberg_marquardt_type::QUADRATIC:
                    lambda = lambda + std::fabs(0.5 * (x2_p - x2_p_new) / alpha_quadratic);
                    break;

                case levenberg_marquardt_type::NIELSEN:
                    lambda *= nu;
                    nu *= 2;
                    break;
                }
            }
        }
    }

    optimization_algorithm_output output(num_residuals_);

    output.update(x2_converged, parameters_converged, gradient_converged, iteration, y_p);

    allocator<scalar_type>::free(data, device_type::CPU);

    return output;
}
}  // namespace quarisma
