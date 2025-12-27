#pragma once

#ifndef __QUARISMA_WRAP__

#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include "MathModule.h"
#include "terminals/matrix.h"

namespace quarisma
{
namespace finite_difference
{

MATH_API void nonuniform_five_points_derivative_0(
    const double* x, size_t i, double*& convection, double*& diffusion);

MATH_API void nonuniform_five_points_derivative_1(
    const double* x, size_t i, double*& convection, double*& diffusion);

MATH_API void nonuniform_five_points_derivative_mid(
    const double* x, size_t i, double*& convection, double*& diffusion, bool upwind);

MATH_API void nonuniform_three_points_derivative_mid(
    const double* x, size_t i, double*& convection, double*& diffusion, bool upwind);

MATH_API void nonuniform_derivation(
    const double* x,
    size_t        nx,
    double*&      convection_ptr,
    double*&      diffusion_ptr,
    bool          upwind,
    size_t        stencil);
};  // namespace finite_difference
}  // namespace quarisma

#endif