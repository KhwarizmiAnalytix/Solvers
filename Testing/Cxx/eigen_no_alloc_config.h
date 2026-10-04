// Force-included (see Testing/Cxx/CMakeLists.txt) into every translation unit of
// the LmNoAllocCheck executable. It turns on Eigen's runtime malloc guard and
// keeps Eigen's assertions live in release builds, where they would otherwise
// compile away and silently disable the guard.
#pragma once

#include <stdexcept>

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (!(x))                                                                                  \
            throw std::runtime_error("Eigen assertion failed: " #x);                               \
    } while (false)
