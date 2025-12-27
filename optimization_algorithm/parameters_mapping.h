#pragma once

#ifndef __QUARISMA_WRAP__

#include <cmath>

#include "common/macros.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace quarisma
{
constexpr double SCALING_FACTOR    = 0.1;
constexpr double ONE_MINUS_EPSILON = 0.999999;

//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double map_to_R(
    const double x,
    const double lower_bound,
    const double upper_bound,
    double       scaling_factor = SCALING_FACTOR)
{
    QUARISMA_CHECK_DEBUG(
        lower_bound <= x && x <= upper_bound, "x must be in the range [lower_bound, upper_bound]");
    if (x > lower_bound && x < upper_bound)
    {
        auto y = (2. * x - upper_bound - lower_bound) / (upper_bound - lower_bound);
        return 0.5 * log((1. + y) / (1. - y)) / scaling_factor;
    }
    else
    {
        return 0.;
    }
}

//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double map_from_R(
    const double y,
    const double lower_bound,
    const double upper_bound,
    double       scaling_factor = SCALING_FACTOR)
{
    return 0.5 *
           (tanh(scaling_factor * y) * (upper_bound - lower_bound) + upper_bound + lower_bound);
}

//-----------------------------------------------------------------------------
QUARISMA_FORCE_INLINE double map_from_R_aad(
    const double ret_aad,
    const double y,
    const double lower_bound,
    const double upper_bound,
    double       scaling_factor = SCALING_FACTOR)
{
    auto x = tanh(scaling_factor * y);
    return ret_aad * 0.5 * scaling_factor * (1. - x * x) * (upper_bound - lower_bound);
}

}  // namespace quarisma

#endif
