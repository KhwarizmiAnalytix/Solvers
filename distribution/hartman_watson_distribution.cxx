#include "distribution/hartman_watson_distribution.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "common/constants.h"
#include "quadrature/gaussian_quadrature.h"
#include "terminals/vector.h"
#include "util/exception.h"

constexpr double error = 25.;

namespace quarisma
{
namespace
{
//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double upper_bound(
    const double r, double t, double max_bound, const double log_epsilon) noexcept
{
    const auto a     = acosh(t * max_bound / r) / t;
    const auto bound = std::min(r < t * max_bound ? std::max(a, 0.002) : 0.002, log_epsilon);

    if (t < 1. && bound < 1.)
    {
        return 1.;
    }

    return bound;
}

//-----------------------------------------------------------------------------
/*
 * kernel of the hartman watson integral
 */
QUARISMA_FORCE_INLINE double kernel(double r, double t, double lb, double x) noexcept
{
    const auto y   = lb * (x + 1.);
    const auto t_y = t * y;

    return exp(-(0.5 * t_y * t_y + r * (cosh(t_y) + 1.)) / t) * sinh(t_y) * sin(constants::PI * y);
}

//-----------------------------------------------------------------------------
//#define DEBUG_KERNEL_INTEGRAL
QUARISMA_FORCE_INLINE double kernel_integral(
    double r, double t, double lb, const vector<double>& roots, const vector<double>& weights)
{
    const auto n = roots.size() - 1;
    lb *= 0.5;

    auto results = weights[n] * kernel(r, t, lb, roots[n]);

    for (size_t i = 0; i < n; i++)
    {
        results += weights[i] * (kernel(r, t, lb, roots[i]) + kernel(r, t, lb, -roots[i]));
    }

    results *= 0.5;

#ifdef DEBUG_KERNEL_INTEGRAL
    if (verbose)
    {
        auto results_tmp = w2[n] * kernel(r, t, lb, x[n]);

        for (size_t i = 0; i < n; i++)
        {
            results_tmp += w2[i] * kernel(r, t, lb, x[i]);
            results_tmp += w2[i] * kernel(r, t, lb, -x[i]);
        }

        const auto error = results_tmp - results;
        std::cout << error << std::endl;
    }
#endif
    return results;
}

//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double first_term_of_the_asymptotic_expansion(const double x, const double tan)
{
    return 0.5 * x * x - x * tan;
}

//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double second_term_of_the_asymptotic_expansion(const double x, const double l2)
{
    return sqrt(l2) / x;
}

//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double third_term_of_the_asymptotic_expansion(
    const double rho, const double l, const double l2)
{
    return (-1. + l * (0.75 - 0.1666666666666666667 * l) + 0.4166666666666666667 * rho * rho) /
           (l2 * l2 * l2);
}
}  // namespace

//-----------------------------------------------------------------------------

//tex:
// $x \in [-\infty,0]$, $\rho =\frac{x}{\sinh(x)}$,
// $x \in ]0,\pi[$, $\rho = \frac{x}{\sin(x)}$, and $r =\frac{\rho}{t}$
// $$logDistributionAsymptotique(x,t) = \theta(r,t)\exp(-r-\frac{t}{8})\frac{\sqrt{2\pi}}{r\sqrt{r}}$$
// $x \in [-\infty,0]$, $$d\rho=\frac{\rho}{x}(1-\rho\cosh(x))dx=-\frac{\rho}{G^2(\rho)} x dx$$  and
// $x \in ]0,\pi[$, $$d\rho=\frac{\rho}{x}(1-\rho\cos(x))dx=\frac{\rho}{G^2(\rho)} x dx$$
// if we consider $x=sign(u)\sqrt{|u|}$
//$$\frac{dr}{r}=\frac{du}{2G^2(\rho)}$$

//-----------------------------------------------------------------------------
double hartman_watson_distribution::log_distribution_asymptotique(double x, double t)
{
    double F;
    double G_inverse;
    double g;
    double rho;

    if (fabs(x) < 0.01)
    {
        const auto x_2 = x * x;
        rho            = 1. + std::copysign(x_2 / 6., x) + 7. / 360 * x_2 * x_2;

        const auto l = x < 0 ? x * tanh(0.5 * x) : -x * tan(0.5 * x);

        F         = -0.5 * std::copysign(x_2, x) - l;
        G_inverse = constants::INVERSE_SQRT_3 *
                    (1 + x_2 * (std::copysign(1. / 30., x) + 11. / 4200. * x_2));
        g = -1. / 35. + 144. / 67375. * (1. / rho - 1.);
    }
    else
    {
        if (x < 0.)
        {
            rho = x / sinh(x);

            const auto l           = rho * cosh(x);
            const auto l_minus_one = l - 1.;

            F         = first_term_of_the_asymptotic_expansion(x, tanh(0.5 * x));
            G_inverse = -second_term_of_the_asymptotic_expansion(x, l_minus_one);
            g         = third_term_of_the_asymptotic_expansion(rho, l, l_minus_one);
        }
        else
        {
            rho = x / sin(x);

            const auto l           = rho * cos(x);
            const auto one_minus_l = 1. - l;

            F         = -first_term_of_the_asymptotic_expansion(x, tan(0.5 * x));
            G_inverse = second_term_of_the_asymptotic_expansion(x, one_minus_l);
            g         = -third_term_of_the_asymptotic_expansion(rho, l, one_minus_l);
        }
    }

    return -(F / t + 0.125 * t) +
           log(G_inverse * (1. + 0.5 * t * g) / (constants::SQRT_2PI * sqrt(rho * t)));
}

//-----------------------------------------------------------------------------
void hartman_watson_distribution::distribution_asymptotic(
    vector<double>& results, const double t, const vector<double>& x)
{
    for (size_t i = 0; i < x.size(); i++)
    {
        results[i] = log_distribution_asymptotique(x[i], t);
    }

    results = exp(results);
}

//-----------------------------------------------------------------------------
double hartman_watson_distribution::distribution_numerical(
    const double          x,
    const double          t,
    const vector<double>& roots,
    const vector<double>& weights,
    const double          log_epsilon,
    const double          shift,
    const double          max_bound)
{
    double drho_dx_over_sqrt_rho;
    double rho;
    if (fabs(x) < 0.001)
    {
        const auto x2 = x * x;
        rho           = 1. + std::copysign(x2 / 6., x) + 7. / 360 * x2 * x2;
        drho_dx_over_sqrt_rho =
            1. / (3. * sqrt(rho)) * (1. + 7. / 30. * std::copysign(x2, x) + 31. / 840. * x2 * x2);
    }
    else if (x < 0)
    {
        rho                   = x / sinh(x);
        drho_dx_over_sqrt_rho = sqrt(rho) * (rho * cosh(x) - 1.) / (x * x);
    }
    else
    {
        rho                   = x / sin(x);
        drho_dx_over_sqrt_rho = sqrt(rho) * (1. - rho * cos(x)) / (x * x);
    }

    const auto lb = upper_bound(rho, t, max_bound, log_epsilon);

    const auto results = kernel_integral(rho, t, lb, roots, weights);

    QUARISMA_CHECK_DEBUG(results >= 0., "result is negatif");

    return shift * drho_dx_over_sqrt_rho * lb * results;
}

//-----------------------------------------------------------------------------
void hartman_watson_distribution::distribution_numerical_integral(
    vector<double>&       results,
    const double          t,
    const vector<double>& x,
    const vector<double>& roots,
    const vector<double>& weights,
    double                max_bound)
{
    const auto cst = constants::PI * constants::PI / (2. * t) -
                     log(constants::PI * constants::SQRT_2PI * sqrt(t));

    if (cst > 0.)
    {
        max_bound += cst;
    }

    const auto log_epsilon = sqrt(2. * max_bound / t) + 1.;
    const auto shift = exp(constants::PI * constants::PI / (2. * t) - 0.125 * t) / constants::PI;

    for (size_t i = 0; i < x.size(); i++)
    {
        results[i] = distribution_numerical(x[i], t, roots, weights, log_epsilon, shift, max_bound);
    }
}

//-----------------------------------------------------------------------------
double hartman_watson_distribution::distribution_numerical(
    const double x, const double t, const vector<double>& roots, const vector<double>& weights)
{
    auto max_bound = 37.0;

    const auto cst = constants::PI * constants::PI / (2. * t) -
                     log(constants::PI * constants::SQRT_2PI * sqrt(t));

    if (cst > 0.)
    {
        max_bound += cst;
    }

    const auto log_epsilon = sqrt(2. * max_bound / t) + 1.;
    const auto shift = exp(constants::PI * constants::PI / (2. * t) - 0.125 * t) / constants::PI;

    return distribution_numerical(x, t, roots, weights, log_epsilon, shift, max_bound);
}

//-----------------------------------------------------------------------------
void hartman_watson_distribution::distribution(
    vector<double>&                        results,
    const double                           t,
    const vector<double>&                  x,
    const vector<double>&                  roots,
    const vector<double>&                  weights,
    const hartman_watson_distribution_enum type,
    double                                 max_bound)
{
    switch (type)
    {
    case hartman_watson_distribution_enum::FULLY_ASYMPTOTIC:
        distribution_asymptotic(results, t, x);
        return;

    case hartman_watson_distribution_enum::NUMERICAL_INTEGRAL:
        distribution_numerical_integral(results, t, x, roots, weights, max_bound);
        return;

    default:
        if (t >= 0.5)
        {
            distribution_numerical_integral(results, t, x, roots, weights, max_bound);
        }
        else
        {
            distribution_asymptotic(results, t, x);
        }
        return;
    }
}

//-----------------------------------------------------------------------------
//tex:
//$$density(z,t)=\sqrt{2\pi}\theta(z,t)\frac{\exp{(-z-\frac{t}{8})}}{z\sqrt{z}}$$
//-----------------------------------------------------------------------------
void hartman_watson_distribution::cheyette_density(
    vector<double>&       density,
    vector<double>&       U,
    double                t,
    const vector<double>& roots,
    const vector<double>& weights,
    const vector<double>& hartman_watson_roots,
    const vector<double>& hartman_watson_weights)
{
    const auto n     = roots.size();
    const auto alpha = (25. - 0.5 * log(t) - 0.125 * t);

    //tex:for $x \to 0$
    //   $$\frac{0.5 x^2 - x  \tanh(\frac{x}{2})}{t} +\log(\frac{\sqrt{x\coth(x)-1}}{x}) =-\frac{\log(3)}{2} -\frac{x^2}{30}+\frac{13x^4}{6300}-\frac{11}{70875}x^6+ \frac{x^4}{24t}(1 - 0.1x^2) + o(x^6).$$
    //   $$\frac{-0.5 x^2 + x  \tan(\frac{x}{2})}{t}+\log(\frac{\sqrt{1-x\cot(x)}}{x}) = -\frac{\log(3)}{2}+\frac{x^2}{30}+\frac{13x^4}{6300}+\frac{11}{70875}x^6+\frac{x^4}{24t}(1 + 0.1x^2) + o(x^6).$$
    // Solve $t \to 0$: $$x^2\sqrt{1+0.049524 t - (0.1+0.003724t)x^2} = \sqrt{24t(25 - \frac{\ln(t)}{2} - \frac{t}{8})}$$  and $$x^2\sqrt{1+0.049524 t + (0.1+0.003724t)x^2} = \sqrt{24t(25 - \frac{\ln(t)}{2} - \frac{t}{8})}$$

    const auto small_t_bound = sqrt(24. * t * (alpha + 0.5 * log(3)));

    double small_t_bound_down;
    double small_t_bound_up;
    {
        const auto a = 0.5 * (0.1 + 0.003724 * t) / (1 + 0.049524 * t);
        const auto b = sqrt(1 + 0.049524 * t);
        const auto c = -small_t_bound;

        small_t_bound_down = (4 * a * small_t_bound < 1.)
                                 ? (-b + sqrt(b * b + 4 * a * c)) / (-2. * a)
                                 : small_t_bound;

        small_t_bound_up = (-b + sqrt(b * b - 4 * a * c)) / (2. * a);
    }

    //tex: for $x \to -\infty$
    double x_min;
    {
        //tex:Solve $$\frac{a}{2}x^2 - bx - c = 0,$$
        //where $a = \frac{1}{t}$, $b = 0.5 + a$ and $c =  25 - \frac{\ln(t)}{2} - \frac{t}{8}$

        const auto a     = 1. / t;
        const auto b     = (0.5 + a);
        const auto delta = b * b + 2. * a * alpha;
        x_min            = (b + sqrt(delta)) / a;
    }

    //tex: for $x \to \pi$, we consider $x=\pi-y$
    //$$-0.5 (\pi-y) ^ 2 + (\pi-y)  \frac{\cos(\frac{y} {2})}{\sin(\frac{y} {2})} = -0.5 (\pi-y) ^ 2 + \frac{(\pi-y)}{y}(2-\frac{y^2}{6}) + o(y ^ 3).$$

    double y_min;
    {
        const auto a     = 2.61799;
        const auto b     = -6.9348 - t * alpha;
        const auto c     = 6.28319;
        const auto delta = b * b - 4 * a * c;

        y_min = delta > 0. ? (-b - sqrt(b * b - 4 * a * c)) / (2. * a) : -c / b;
    }

    const auto to   = t <= 0.1 ? small_t_bound_up : std::sqr(constants::PI - y_min);
    const auto from = t <= 0.08 ? -small_t_bound_down : -std::sqr(x_min);

    const auto mid_diff    = 0.5 * (to - from);
    const auto mid_average = 0.5 * (to + from);
    const auto offset      = 2 * n - 2;

    for (size_t i = 0; i < n - 1; ++i)
    {
        U[i]          = -mid_diff * roots[i] + mid_average;
        U[offset - i] = mid_diff * roots[i] + mid_average;
    }

    U[n - 1] = -mid_diff * roots[n - 1] + mid_average;

    U = if_else(U < 0., -sqrt(-U), sqrt(U));

    quarisma::hartman_watson_distribution::distribution(
        density,
        t,
        U,
        hartman_watson_roots,
        hartman_watson_weights,
        quarisma::hartman_watson_distribution_enum::MIXTURE);

    auto cst1 = 0.5 * mid_diff;

    density *= cst1 * weights;

    U = if_else(U < 0, U / sinh(U), if_else(U == 0., 1., U / sin(U)));
    U /= t;
}
}  // namespace quarisma
