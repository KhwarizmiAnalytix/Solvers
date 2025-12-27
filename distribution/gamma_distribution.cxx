#include "distribution/gamma_distribution.h"

#include <array>
#include <cmath>

#include "common/constants.h"

namespace quarisma
{
namespace
{
//-----------------------------------------------------------------------------
std::array<double, 9> cof = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7};

//-----------------------------------------------------------------------------
template <typename T>
T gammaln(T x)
{
    auto y = x + 7.5;
    y -= (x + 0.5) * std::log(y);

    auto fraction = cof[0] + cof[1] / (1. + x) + cof[2] / (2. + x) + cof[3] / (3. + x) +
                    cof[4] / (4. + x) + cof[5] / (5. + x) + cof[6] / (6. + x) + cof[7] / (7. + x) +
                    cof[8] / (8. + x);

    return log(constants::SQRT_2PI * fraction / x) - y;
}
}  // namespace

//-----------------------------------------------------------------------------
double gamma_distribution::lgamma(double x)
{
    if (x < 0.5)
    {
        auto z = quarisma::constants::PI / sin(-quarisma::constants::PI * x);
        return std::log(std::fabs(z)) - gammaln(1. - x);
    }

    return gammaln(x);
}

//-----------------------------------------------------------------------------
double gamma_distribution::gamma(double x)
{
    return exp(lgamma(x));
}

//-----------------------------------------------------------------------------
std::complex<double> gamma_distribution::lgamma(const std::complex<double>& x)
{
    if (x.real() < 0.5)
    {
        const auto z = quarisma::constants::PI / sin(-quarisma::constants::PI * x);
        return std::log(z) - gammaln(1. - x);
    }
    return gammaln(x);
}
}  // namespace quarisma
