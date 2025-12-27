#include "distribution/normal_distribution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "common/constants.h"
#include "common/normal_cdf.h"

namespace quarisma
{
//-----------------------------------------------------------------------------
const double nor_cdf_asymptotic_expansion_threshold =
    -1 / std::sqrt(std::numeric_limits<double>::epsilon());

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_2(T1 coefficients, T2 X)
{
    return (X * X * coefficients[2] + X * coefficients[1] + coefficients[0]);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_3(T1 coefficients, T2 X)
{
    return (coefficients[0] + (coefficients[1] + (coefficients[2] + coefficients[3] * X) * X) * X);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_4(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] + (coefficients[2] + (coefficients[3] + coefficients[4] * X) * X) * X) *
            X);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_5(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] +
         (coefficients[2] + (coefficients[3] + (coefficients[4] + coefficients[5] * X) * X) * X) *
             X) *
            X);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_6(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] +
         (coefficients[2] +
          (coefficients[3] + (coefficients[4] + (coefficients[5] + coefficients[6] * X) * X) * X) *
              X) *
             X) *
            X);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_7(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] +
         (coefficients[2] +
          (coefficients[3] +
           (coefficients[4] + (coefficients[5] + (coefficients[6] + coefficients[7] * X) * X) * X) *
               X) *
              X) *
             X) *
            X);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_8(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] +
         (coefficients[2] +
          (coefficients[3] +
           (coefficients[4] +
            (coefficients[5] +
             (coefficients[6] + (coefficients[7] + coefficients[8] * X) * X) * X) *
                X) *
               X) *
              X) *
             X) *
            X);
}

//-----------------------------------------------------------------------------
double normal_distribution::density(double x)
{
    return exp(-0.5 * x * x) * quarisma::constants::INVERSE_SQRT_2PI;
}

//-----------------------------------------------------------------------------
double normal_distribution::cdf_over_density(double z)
{
    auto   x = std::fabs(z);
    double p = 0.;

    if (x < 7.071067811865475)
    {
        static std::array<double, 7> P = {
            220.2068679123761,
            221.2135961699311,
            112.0792914978709,
            33.912866078383,
            6.37396220353165,
            .7003830644436881,
            .03526249659989109};

        static std::array<double, 8> Q = {
            440.4137358247522,
            793.8265125199484,
            637.3336333788311,
            296.5642487796737,
            86.78073220294608,
            16.06417757920695,
            1.755667163182642,
            .08838834764831844};

        p = constants::SQRT_2PI * POLYNOM_6(P, x) / POLYNOM_7(Q, x);
    }
    else
    {
        p = 1. / (x + 1. / (x + 2. / (x + 3. / (x + 4. / (x + .65)))));
    }

    return (z > 0.) ? exp(0.5 * x * x) * quarisma::constants::SQRT_2PI - p : p;
}

//-----------------------------------------------------------------------------
// Based upon
// algorithm 5666 for the error function, from:
/// Hart, J.Q3. et al, 'Computer Approximations', Wiley 1968
//-----------------------------------------------------------------------------
double normal_distribution::cdf(double z)
{
    return quarisma::normalcdf(z);
}

//-----------------------------------------------------------------------------
// Peter J. Acklam Method.
// URL: http://www.math.uio.no/~jacklam/notes/invnorm
//-----------------------------------------------------------------------------
double normal_distribution::inv_cdf_fast(double p)
{
    p = p - 0.5;

    if (std::fabs(p) <= .47575)
    {
        static std::array<double, 5> P1 = {
            1.77792631017149349581529804709135,
            -20.98714019962876875524671049788594,
            89.67951420989056998678279342129827,
            -162.47039264349248810503922868520021,
            103.19822664969072434359986800700426};

        static std::array<double, 6> Q1 = {
            1.,
            -13.28068206353032233422245,
            66.80131811358923018209895,
            -155.6990071150346771962174,
            161.5858866432906228265978,
            -54.47612957075147289732531};

        const auto x = p * p;

        return p * (POLYNOM_4(P1, x) / POLYNOM_5(Q1, x) + 0.72870196728882574710439712362131);
    }
    static std::array<double, 4> P2 = {
        2.92803064137654889265149904531427,
        5.33664490923877465888836013618857,
        1.17999437685196673086807095387485,
        0.04117047346258065343427290372347,
    };

    static std::array<double, 5> Q2 = {
        1., 3.754408661907416, 2.445134137142996, 3.224671290700398e-01, 7.784695709041462e-03};

    auto z = std::sqrt(-2. * std::log(.5 - std::fabs(p)));

    z = -1.00002547220806592420672131993342 * z + 0.01013334132223406985895408638498 +
        POLYNOM_3(P2, z) / POLYNOM_4(Q2, z);

    return p < 0. ? z : -z;
}

//-----------------------------------------------------------------------------
// The algorithm in Wichura's 1988 paper on ALGORITHM AS241 APPL. STATIST.
// (1988) VOL. 37, NO. 3 Produces the normal_distribution deviate Z
// corresponding to a given lower tail area of P; Z is accurate to about 1 part
// in 10**16.
//-----------------------------------------------------------------------------
double normal_distribution::inv_cdf(double p)
{
    return quarisma::inv_normalcdf(p);
}

//-----------------------------------------------------------------------------
void normal_distribution::inv_cdf(size_t size, const double* uniform, double* gaussian)
{
    std::transform(uniform, uniform + size, gaussian, inv_cdf_fast);
}
}  // namespace quarisma
