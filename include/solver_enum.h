#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "include/detail/support.h"

namespace solverslib
{
enum class solver_enum : int
{
    LM           = 0,
    LBFGS        = 1,
    NLOPT        = 2,
    CERES        = 3,
    GAUSS_NEWTON = 4
};
}  // namespace solverslib
