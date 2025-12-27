#pragma once

namespace quarisma
{
enum class interpolation_enum : int
{
    LINEAR                   = 0,
    CUBIC_HERMITE            = 1,
    CUBIC_SPLINE             = 2,
    PIECEWISE_CONSTANT_LEFT  = 3,
    PIECEWISE_CONSTANT_RIGHT = 4,
    LINEAR_EXPONENTIAL       = 5,
    GEOMETRIC                = 6,
    GEOMETRIC_AVERAGE        = 7,
    AVERAGE                  = 8,

    //Volatility interpolators
    MEAN_REVERTING = 7,
    FIXED_STRIKE   = 8,
    // Used for inflation data interpolation
    // Custom interpolator for bespoke cases
    BESPOKE = 9
};
}  // namespace quarisma