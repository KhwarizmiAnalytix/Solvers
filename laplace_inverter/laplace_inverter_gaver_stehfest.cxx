#include "laplace_inverter/laplace_inverter_gaver_stehfest.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "common/constants.h"

namespace quarisma
{
namespace
{
constexpr double _log2 = constants::LOG_2;

double int_pow(size_t x, size_t p)
{
    if (p == 0)
    {
        return 1;
    }
    if (p == 1)
    {
        return (double)x;
    }

    double tmp = int_pow(x, p / 2);
    return (p % 2 == 0) ? tmp * tmp : ((double)x) * tmp * tmp;
}

double factorial(size_t k)
{
    double fact = 1;

    if (k == 0 || k == 1)
    {
        return fact;
    }
    for (size_t i = 1; i <= k; i++)
    {
        fact *= (double)i;
    }

    return fact;
}
}  // namespace

//-----------------------------------------------------------------------------
laplace_inverter_gaver_stehfest::laplace_inverter_gaver_stehfest(size_t size, double shift)
    : shift_(shift), coefficient_(2 * size)
{
    const auto twice_size = 2 * size;
    for (size_t i = 0; i < twice_size; i++)
    {
        double z = 0.;

        for (size_t j = (i + 2) / 2; j <= std::min(i + 1, size); j++)
        {
            z += ((int_pow(j, size)) * factorial(2 * j)) /
                 (factorial(size - j) * factorial(j) * factorial(j - 1) * factorial(i + 1 - j) *
                  factorial(2 * j - i - 1));
        }

        coefficient_[i] = (i + 1 + size) % 2 == 0 ? z : -z;
    }
}

//-----------------------------------------------------------------------------
double laplace_inverter_gaver_stehfest::operator()(
    double (*F)(double, double), double t, double r) const
{
    double _log2_t = _log2 / t;

    double sum = 0.;
    for (size_t i = 0, size = coefficient_.size(); i < size; i++)
    {
        sum += coefficient_[i] * F(shift_ + (double)(i + 1) * _log2_t, r);
    }

    return exp(shift_ * t) * sum * _log2_t;
}

}  // namespace quarisma
