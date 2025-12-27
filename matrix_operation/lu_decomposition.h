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
MATH_API bool lu_decomposition(float* m, quarisma_int lda, quarisma_int* pivot);

MATH_API bool lu_decomposition(double* m, quarisma_int lda, quarisma_int* pivot);

}  // namespace quarisma
#endif