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
enum class linear_solver_type : quarisma_int
{
    LU_LINEAR_SOLVER               = 1,
    LU_UPFRONT_LINEAR_SOLVER       = 2,
    CHOLESKY_LINEAR_SOLVER         = 3,
    CHOLESKY_UPFRONT_LINEAR_SOLVER = 4,
};
#endif

MATH_API void linear_solver(
    float*             m,
    quarisma_int*        pivot,
    quarisma_int         lda,
    float*             x,
    linear_solver_type type = linear_solver_type::LU_LINEAR_SOLVER);

MATH_API void linear_solver(
    double*            m,
    quarisma_int*        pivot,
    quarisma_int         lda,
    double*            x,
    linear_solver_type type = linear_solver_type::LU_LINEAR_SOLVER);

}  // namespace quarisma
