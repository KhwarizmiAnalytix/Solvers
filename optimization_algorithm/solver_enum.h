#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "MathModule.h"

namespace quarisma
{
enum class solver_enum : int
{
    LM    = 0,
    LBFGS = 1,
    NLOPT = 2,
    CERES = 3
};
}  // namespace quarisma