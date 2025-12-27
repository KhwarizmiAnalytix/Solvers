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
namespace pentadiagonal_operations
{

MATH_API void decomposition(
    matrix<double>& output,
    const size_t    outer_dim,
    const size_t    dim,
    const size_t    inner_dim,
    const bool      parallelize = false);

MATH_API void solve_decomposed(
    vector<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize = false);

MATH_API void multiply(
    vector<double>&       result,
    const vector<double>& in,
    const matrix<double>& mat,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    double                time_multiplier,
    const bool            parallelize = false);
};  // namespace pentadiagonal_operations
}  // namespace quarisma

#endif