#include "distribution/inverse_gaussian_distribution.h"

#include <cmath>

#include "common/constants.h"
#include "distribution/normal_distribution.h"

namespace quarisma
{
//-----------------------------------------------------------------------------
double inverse_gaussian_distribution::density(double x, double lambda)
{
    const auto y = (x - 1.);
    return exp(-lambda * y * y / (2. * x)) * sqrt(lambda / x) / (constants::SQRT_2PI * x);
}

//-----------------------------------------------------------------------------
double inverse_gaussian_distribution::cdf(double x, double lambda)
{
    if (is_almost_zero(x))
    {
        return 0.;
    }

    const auto a = -sqrt(lambda / x);

    const auto x_minus = a * (1. - x);
    const auto x_plus  = a * (1. + x);

    auto n = normal_distribution::cdf(x_plus);

    bool non_zero = n > 0.;

    if (non_zero)
    {
        n = log(n);
    }

    return normal_distribution::cdf(x_minus) + (non_zero ? exp(2. * lambda + n) : 0.);
}
}  // namespace quarisma
