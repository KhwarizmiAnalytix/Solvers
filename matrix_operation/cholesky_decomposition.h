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

namespace quarisma
{
#ifndef __QUARISMA_WRAP__

enum class cholesky_decomposition_enum : char
{
    LOWER_TRIANGULAR = 'L',
    UPPER_TRIANGULAR = 'U'
};

MATH_API bool cholesky_decomposition(
    float* L, quarisma_int lda, quarisma::cholesky_decomposition_enum type);

MATH_API bool cholesky_decomposition_aad(
    float*                              L_aad,
    const float*                        L,
    quarisma_int                          lda,
    quarisma::cholesky_decomposition_enum type,
    float*                              A_aad);

MATH_API bool cholesky_decomposition(
    double* L, quarisma_int lda, quarisma::cholesky_decomposition_enum type);

MATH_API bool cholesky_decomposition_aad(
    double*                             L_aad,
    const double*                       L,
    quarisma_int                          lda,
    quarisma::cholesky_decomposition_enum type,
    double*                             A_aad);

#endif
}  // namespace quarisma
