#include "common/karhunen_loeve_expansion.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "common/constants.h"
#include "expressions/expressions.h"
#include "terminals/matrix.h"
#include "terminals/vector.h"

namespace quarisma
{
namespace
{
//-----------------------------------------------------------------------------
double root_finding_algorithm(
    const double mean_reversion_speed,
    const double tau,
    size_t       n             = 0,
    short        max_iteration = 20,
    const double tolerance     = 1.E-10)
{
    const double a   = (mean_reversion_speed * tau);
    const double max = (static_cast<double>(n) + 1.) * quarisma::constants::PI - tolerance;
    const double min = (static_cast<double>(n) + .5) * quarisma::constants::PI + tolerance;

    auto x = 0.1 * max + 0.9 * min;
    auto f = (a * std::sin(x) + x * std::cos(x));

    short iter = 0;
    while (iter++ < max_iteration && std::fabs(f) > tolerance)
    {
        const auto cos_x = std::cos(x);
        const auto sin_x = std::sin(x);

        f             = (a * sin_x + x * cos_x);
        const auto f1 = (cos_x * (1. + a) - x * sin_x);
        const auto f2 = -(sin_x * (a + 2.) + x * cos_x);

        const auto dx = f * f1 / (f1 * f1 - 0.5 * f * f2);

        if (quarisma::is_almost_zero(dx))
        {
            break;
        }

        x = std::clamp(x - dx, min, max);
    }
    return x / tau;
}

//-----------------------------------------------------------------------------
double variance(
    size_t                     start,
    size_t                     end,
    double                     t,
    double                     T,
    const std::vector<double>& variance,
    const std::vector<double>& dates)
{
    double sum = 0.;

    auto from = t;
    for (size_t i = start + 1; i <= end; ++i)
    {
        const auto to = dates[i];

        sum += (to - from) * variance[i - 1];
        from = to;
    }

    sum += (T - from) * variance[end];

    return sum / (T - t);
}

//-----------------------------------------------------------------------------
double root_finder_algorithm(
    std::function<void(double, double&, double&)> const& objectiv_function,
    const double                                         mean_reversion_speed,
    size_t                                               start,
    size_t                                               end,
    const double                                         t,
    const double                                         T,
    double                                               omega,
    const std::vector<double>&                           variances,
    const std::vector<double>&                           dates,
    short                                                max_iteration = 20,
    const double                                         tolerance     = 1.E-10)
{
    if (start == end)
    {
        return variances[start] / ((omega * omega + mean_reversion_speed * mean_reversion_speed));
    }

    const auto var_avrg = variance(start, end, t, T, variances, dates);

    auto itr_start = variances.begin();
    auto itr_end   = variances.begin();
    itr_start += start;  //NOLINT
    itr_end += end;      //NOLINT

    const auto lower_bound = *std::min_element(itr_start, itr_end) /
                             (omega * omega + mean_reversion_speed * mean_reversion_speed);

    auto x = var_avrg / ((omega * omega + mean_reversion_speed * mean_reversion_speed));

    const auto upper_bound = *std::max_element(itr_start, itr_end) /
                             (omega * omega + mean_reversion_speed * mean_reversion_speed);

    double f;
    double f_aad;
    objectiv_function(x, f, f_aad);

    short iter = 0;
    while (iter++ < max_iteration && std::fabs(f) > tolerance)
    {
        const auto dx = f / f_aad;

        if (quarisma::is_almost_zero(dx))
        {
            break;
        }

        x = std::clamp(x - dx, lower_bound, upper_bound);

        objectiv_function(x, f, f_aad);
    }

    return x;
}

//-----------------------------------------------------------------------------
double integral_norm_trigonometric(double a, double b, double omega, double tau)
{
    return (2 * omega * tau * (a * a + b * b) + (a * a - b * b) * sin(2 * omega * tau) +
            2 * a * b * (1 - cos(2 * omega * tau))) /
           (4 * omega);
}

//-----------------------------------------------------------------------------
double integral_norm_hyperbolic(double a, double b, double omega, double tau)
{
    return (2 * omega * tau * (a * a - b * b) + (a * a + b * b) * sin(2 * omega * tau) +
            2 * a * b * (cosh(2 * omega * tau) - 1)) /
           (4 * omega);
}

//-----------------------------------------------------------------------------
double norm_l2(
    quarisma::vector<double>&    A,
    quarisma::vector<double>&    B,
    size_t                     start,
    size_t                     end,
    double                     t,
    double                     T,
    double                     lambda,
    double                     mean_reversion_speed,
    const std::vector<double>& variance,
    const std::vector<double>& dates)
{
    const auto kappa_2 = mean_reversion_speed * mean_reversion_speed;

    size_t k = 0;
    size_t i = start + 1;

    auto omega_2 = variance[start] / lambda - kappa_2;
    auto omega   = sqrt(std::fabs(omega_2));

    double norm = 0.;
    auto   from = t;
    for (; i <= end; ++i)
    {
        const auto to = dates[i];

        const auto tau = (to - from);

        norm += omega_2 > 0. ? integral_norm_trigonometric(A[k], B[k], omega, tau)
                             : integral_norm_hyperbolic(A[k], B[k], omega, tau);

        omega_2 = variance[i] / lambda - kappa_2;
        omega   = sqrt(std::fabs(omega_2));
        ++k;
        from = to;
    }
    const auto tau = (T - from);

    norm += omega_2 > 0. ? integral_norm_trigonometric(A[k], B[k], omega, tau)
                         : integral_norm_hyperbolic(A[k], B[k], omega, tau);

    return norm;
}

//-----------------------------------------------------------------------------
void function_and_gradient(
    double&                    f,
    double&                    f_aad,
    quarisma::vector<double>&    A,
    quarisma::vector<double>&    B,
    size_t                     start,
    size_t                     end,
    double                     t,
    double                     T,
    double                     lambda,
    double                     mean_reversion_speed,
    const std::vector<double>& variance,
    const std::vector<double>& dates)
{
    f_aad = 0.;

    const auto kappa_2 = mean_reversion_speed * mean_reversion_speed;

    A[0] = 0;
    B[0] = 1.;

    auto omega_2 = variance[start] / lambda - kappa_2;
    auto omega   = sqrt(std::fabs(omega_2));

    auto alpha  = std::numeric_limits<double>::quiet_NaN();
    auto beta   = std::numeric_limits<double>::quiet_NaN();
    auto factor = std::numeric_limits<double>::quiet_NaN();
    auto tau    = std::numeric_limits<double>::quiet_NaN();

    size_t k    = 1;
    size_t i    = start + 1;
    auto   from = t;
    for (; i <= end; ++i)
    {
        const auto to = dates[i];

        tau = (to - from);

        alpha  = omega_2 > 0. ? cos(omega * tau) : cosh(omega * tau);
        beta   = omega_2 > 0. ? sin(omega * tau) : sinh(omega * tau);
        factor = omega_2 > 0. ? -1. : 1.;

        omega_2         = variance[i] / lambda - kappa_2;
        auto omega_next = sqrt(std::fabs(omega_2));

        A[k] = A[k - 1] * alpha + B[k - 1] * beta;

        B[k] = (variance[i] / variance[i - 1] *
                    (omega * (factor * A[k - 1] * beta + B[k - 1] * alpha) +
                     mean_reversion_speed * A[k]) -
                mean_reversion_speed * A[k]) /
               omega_next;

        omega = omega_next;
        ++k;
        from = to;
    }
    k--;

    tau   = (T - dates[end]);
    alpha = omega_2 > 0 ? cos(omega * tau) : cosh(omega * tau);
    beta  = omega_2 > 0 ? sin(omega * tau) : sinh(omega * tau);

    const auto F_prim = (factor * A[k] * beta + B[k] * alpha);
    const auto F      = (A[k] * alpha + B[k] * beta);

    f = omega * F_prim + mean_reversion_speed * F;

    auto A_aad     = mean_reversion_speed * alpha + omega * factor * beta;
    auto B_aad     = mean_reversion_speed * beta + omega * alpha;
    auto omega_aad = tau * (mean_reversion_speed * F_prim + factor * omega * F) + F_prim;

    auto to = dates[end];

    for (i = end; i > start; --i, --k)
    {
        from = i == start + 1 ? t : dates[i - 1];

        tau = (to - from);

        const auto omega_prev_2 = variance[i - 1] / lambda - kappa_2;
        const auto omega_prev   = sqrt(std::fabs(omega_prev_2));

        alpha  = omega_2 > 0. ? cos(omega * tau) : cosh(omega * tau);
        beta   = omega_2 > 0. ? sin(omega * tau) : sinh(omega * tau);
        factor = omega_prev_2 > 0 ? -1. : 1.;

        auto tmp_aad = B_aad * variance[i] * omega_prev / (variance[i - 1] * omega);

        //B[k]
        auto A_prev_aad = tmp_aad * factor * beta;
        auto B_prev_aad = tmp_aad * alpha;

        A_aad += B_aad * (variance[i] / variance[i - 1] - 1.) * mean_reversion_speed / omega;

        omega_aad -= B_aad * B[k] / omega;

        f_aad += omega_2 > 0. ? -0.5 * omega_aad * variance[i] / (omega * lambda * lambda)
                              : 0.5 * omega_aad * variance[i] / (omega * lambda * lambda);

        const auto tmp = (factor * A[k - 1] * beta + B[k - 1] * alpha);

        omega_aad = B_aad * variance[i] / variance[i - 1] *
                    (factor * omega_prev * tau * A[k] + tmp) / omega;

        //A[k] = A[k - 1] * alpha + B[k - 1] * beta;
        A_prev_aad += A_aad * alpha;
        B_prev_aad += A_aad * beta;
        omega_aad += A_aad * tau * tmp;

        A_aad = A_prev_aad;
        B_aad = B_prev_aad;

        omega   = omega_prev;
        omega_2 = omega_prev_2;

        to = from;
    }

    f_aad += omega_2 > 0 ? -0.5 * omega_aad * variance[start] / (omega * lambda * lambda)
                         : 0.5 * omega_aad * variance[start] / (omega * lambda * lambda);
}

//-----------------------------------------------------------------------------
inline double norm(const double mean_reversion_speed, double tau, const double omega)
{
    const auto lambda = 1. / (omega * omega + mean_reversion_speed * mean_reversion_speed);

    tau += mean_reversion_speed * lambda;

    return sqrt(2. * lambda / tau);
}

//-----------------------------------------------------------------------------
inline double incremental_variance(
    double var_i, double from, double to, double mean_reversion_speed)
{
    return var_i * (is_almost_zero(mean_reversion_speed)
                        ? to - from
                        : (exp(mean_reversion_speed * to) - exp(mean_reversion_speed * from)) /
                              (mean_reversion_speed));
}

//-----------------------------------------------------------------------------
void integral(
    const quarisma::vector<double>& x,
    const double                  vol,
    const double                  dt,
    const std::vector<double>&    mean_t,
    const std::vector<double>&    std_t,
    const std::vector<double>&    rho_t,
    const double                  z,
    quarisma::vector<double>&       output)
{
    size_t nt = mean_t.size() - 1;

    output = 0.5 * (exp(mean_t[0] + std_t[0] * z + (rho_t[0] * vol) * x) +
                    exp(mean_t[nt] + std_t[nt] * z + (rho_t[nt] * vol) * x));

    for (size_t t = 1; t < nt; ++t)
    {
        output += exp(mean_t[t] + std_t[t] * z + (rho_t[t] * vol) * x);
    }

    output *= -dt;
}

size_t lower_bound(double t, const std::vector<double>& model_dates)
{
    if (t <= model_dates.front())
    {
        return 0UL;
    }

    if (t >= model_dates.back())
    {
        return model_dates.size() - 1;
    }

    const auto offset = static_cast<size_t>(
        std::lower_bound(model_dates.begin(), model_dates.end(), t) - model_dates.begin());

    return offset;
}
}  // namespace

//-----------------------------------------------------------------------------
karhunen_loeve_expansion::karhunen_loeve_expansion(
    double                     from,
    double                     to,
    const double               mean_reversion_speed,
    const std::vector<double>& mean_reversion_levels,
    const std::vector<double>& level_dates,
    const std::vector<double>& variances,
    const std::vector<double>& volatility_dates,
    double                     discretization_step)
{
    assert(from <= to);

    if (from == to)
    {
        define_ = 0;
    }
    else
    {
        const auto tau = to - from;
        const auto nt  = static_cast<size_t>(tau / discretization_step) + 1;
        dt_            = nt == 1 ? tau : tau / (static_cast<double>(nt) - 1.);

        auto start = std::distance(
                         volatility_dates.begin(),
                         std::upper_bound(volatility_dates.begin(), volatility_dates.end(), from)) -
                     1;
        auto end = std::distance(
                       volatility_dates.begin(),
                       std::upper_bound(volatility_dates.begin(), volatility_dates.end(), to)) -
                   1;

        vector<double> A(end - start + 1);
        vector<double> B(end - start + 1);

        auto objectiv_function =
            [start, end, from, to, mean_reversion_speed, &variances, &volatility_dates, &A, &B](
                double x, double& f, double& f_aad)
        {
            function_and_gradient(
                f,
                f_aad,
                A,
                B,
                start,
                end,
                from,
                to,
                x,
                mean_reversion_speed,
                variances,
                volatility_dates);
        };

        auto omega   = root_finding_algorithm(mean_reversion_speed, tau);
        auto omega_2 = omega * omega;
        auto lambda  = root_finder_algorithm(
            objectiv_function,
            mean_reversion_speed,
            start,
            end,
            from,
            to,
            omega,
            variances,
            volatility_dates);

        if (start < end)
        {
            double adjustement = sqrt(
                lambda / norm_l2(
                             A,
                             B,
                             start,
                             end,
                             from,
                             to,
                             lambda,
                             mean_reversion_speed,
                             variances,
                             volatility_dates));

            A *= adjustement;
            B *= adjustement;
            omega_2 = variances[start] / lambda - mean_reversion_speed * mean_reversion_speed;
            omega   = sqrt(std::fabs(omega_2));
        }
        else
        {
            lambda = sqrt(variances[start]) * norm(mean_reversion_speed, tau, omega);
        }

        auto eigenfunction = [start, end, lambda, &A, &B](
                                 size_t k, double t, double t_k, double omega, double omega_2)
        {
            if (start < end)
            {
                return omega_2 > 0.
                           ? A[k] * cos(omega * (t - t_k)) + B[k] * sin(omega * (t - t_k))
                           : A[k] * cosh(omega * (t - t_k)) + B[k] * sinh(omega * (t - t_k));
            }

            return lambda * sin(omega * (t - t_k));
        };

        rho_.resize(nt);
        means_.resize(nt);
        stdev_.resize(nt);

        vector<double> time_steps(nt);
        vector<double> rho(rho_);
        vector<double> means(means_);
        vector<double> stdev(stdev_);

        const auto level_size = level_dates.size();
        const auto vol_size   = volatility_dates.size();

        auto level_offset = lower_bound(from, level_dates);
        auto vol_offset   = lower_bound(from, volatility_dates);

        if (volatility_dates[vol_offset] > from)
        {
            --vol_offset;
        }

        double u   = from;
        double var = 0.;
        size_t k   = 0;
        auto   t_k = from;

        auto time_step_prev = 0.;
        for (size_t i = 0; i < nt; ++i)
        {
            if (u > to || i == nt - 1)
            {
                u = to;
            }

            time_steps[i] = u - from;

            if (level_offset < level_size - 1 && u > level_dates[level_offset])
            {
                level_offset++;
            }

            if (vol_offset < vol_size - 1 && u >= volatility_dates[vol_offset + 1])
            {
                t_k                = volatility_dates[vol_offset + 1];
                const auto tau_vol = t_k - from;

                var += incremental_variance(
                    variances[vol_offset], time_step_prev, tau_vol, 2. * mean_reversion_speed);

                vol_offset++;
                ++k;

                var += incremental_variance(
                    variances[vol_offset], tau_vol, time_steps[i], 2. * mean_reversion_speed);

                omega_2 =
                    variances[vol_offset] / lambda - mean_reversion_speed * mean_reversion_speed;
                omega = sqrt(std::fabs(omega_2));
            }
            else
            {
                var += incremental_variance(
                    variances[vol_offset],
                    time_step_prev,
                    time_steps[i],
                    2. * mean_reversion_speed);
            }

            means_[i] = mean_reversion_levels[level_offset] +
                        0.5 * exp(-2. * mean_reversion_speed * time_steps[i]) * var;

            stdev_[i] = eigenfunction(k, u, t_k, omega, omega_2);
            u += dt_;
            time_step_prev = time_steps[i];
        }

        rho = exp(-mean_reversion_speed * time_steps);
        means -= 0.5 * stdev * stdev;

        define_ = 1;
    }
}

//-----------------------------------------------------------------------------
void karhunen_loeve_expansion::survival_probability(
    const vector<double>& state_variable,
    double                decay,
    const vector<double>& roots,
    const vector<double>& weights,
    vector<double>&       output) const
{
    if (define_ == 0)
    {
        output = 1.;
        return;
    }

    size_t num_of_roots = roots.size();
    size_t n            = output.size();

    vector<double> tmp(n);
    output = 0.;

    for (size_t i = 0; i < num_of_roots; ++i)
    {
        integral(state_variable, decay, dt_, means_, stdev_, rho_, roots[i], tmp);
        output += weights[i] * exp(tmp);
    }
}
}  // namespace quarisma
