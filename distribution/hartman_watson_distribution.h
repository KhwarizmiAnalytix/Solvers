#pragma once

#include <cstddef>

#include "MathModule.h"
#include "common/macros.h"

namespace quarisma
{
template <typename value_t>
class vector;
}

namespace quarisma
{
constexpr double HARTMAN_WATSON_DISTRIBUTION_MAX_BOUND = 37.0;

enum class hartman_watson_distribution_enum : int
{
    FULLY_ASYMPTOTIC   = 1,
    NUMERICAL_INTEGRAL = 2,
    MIXTURE            = 3
};

class hartman_watson_distribution
{
public:
    MATH_API static void distribution(
        vector<double>&                  results,
        double                           t,
        const vector<double>&            x,
        const vector<double>&            roots,
        const vector<double>&            weights,
        hartman_watson_distribution_enum type      = hartman_watson_distribution_enum::MIXTURE,
        double                           max_bound = HARTMAN_WATSON_DISTRIBUTION_MAX_BOUND);

    MATH_API static void cheyette_density(
        vector<double>&       density,
        vector<double>&       U,
        double                t,
        const vector<double>& roots,
        const vector<double>& weights,
        const vector<double>& hartman_watson_roots,
        const vector<double>& hartman_watson_weights);

#ifndef __QUARISMA_WRAP__
    /**
    //tex:
    //$$x \in ]-\infty,0],\: \rho =\frac{x}{\sinh(x)}\\ \: x \in ]0,\pi[,\:\rho = \frac{x}{\sin(x)}$$
    //$$logDistributionAsymptotique(x,t) = \theta(r,t)\exp(-r-\frac{t}{8})\frac{\sqrt{2\pi}}{r\sqrt{r}}, \:  r =\frac{\rho}{t}$$
    **/
#endif
    MATH_API static double log_distribution_asymptotique(double x, double t);

    MATH_API static double distribution_numerical(
        const double x, const double t, const vector<double>& roots, const vector<double>& weights);

private:
    QUARISMA_DELETE_CLASS(hartman_watson_distribution);

    static double distribution_numerical(
        const double          x,
        const double          t,
        const vector<double>& roots,
        const vector<double>& weights,
        const double          log_epsilon,
        const double          log_shift,
        const double          max_bound = HARTMAN_WATSON_DISTRIBUTION_MAX_BOUND);

    static void distribution_asymptotic(
        vector<double>& results, const double t, const vector<double>& x);

    static void distribution_numerical_integral(
        vector<double>&       results,
        const double          t,
        const vector<double>& x,
        const vector<double>& roots,
        const vector<double>& weights,
        double                max_bound);
};
}  // namespace quarisma
