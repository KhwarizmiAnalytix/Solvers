#pragma once

#include <cstddef>

#include "MathModule.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

#ifndef __QUARISMA_WRAP__
namespace quarisma
{
MATH_API void matrix_transpose(quarisma_long rows, quarisma_long columns, float* m);

MATH_API void matrix_transpose(quarisma_long rows, quarisma_long columns, double* m);

}  // namespace quarisma
#endif