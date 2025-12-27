#pragma once

#include <cstddef>

#include "MathModule.h"
#include "matrix_operation/linear_solver.h"

#ifndef __QUARISMA_WRAP__
namespace quarisma
{
MATH_API void matrix_invert(
    float*                     m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    quarisma::linear_solver_type type = quarisma::linear_solver_type::LU_LINEAR_SOLVER);

MATH_API void matrix_invert(
    double*                    m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    quarisma::linear_solver_type type = quarisma::linear_solver_type::LU_LINEAR_SOLVER);

MATH_API float matrix_determinant(
    float*                     m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    quarisma::linear_solver_type type = quarisma::linear_solver_type::LU_LINEAR_SOLVER);

MATH_API double matrix_determinant(
    double*                    m,
    quarisma_int*                pivot,
    quarisma_int                 lda,
    quarisma::linear_solver_type type = quarisma::linear_solver_type::LU_LINEAR_SOLVER);

}  // namespace quarisma
#endif