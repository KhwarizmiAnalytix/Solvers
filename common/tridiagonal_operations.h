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
namespace tridiagonal_operations
{

/**
 * @brief Perform decomposition on the matrix to separate it into parts.
 * 
 * @param output The output matrix after decomposition.
 * @param outer_dim The dimension of the outer part.
 * @param dim The dimension of the matrix.
 * @param inner_dim The dimension of the inner part.
 * @param parallelize Flag to indicate if decomposition should be parallelized.
 */
MATH_API void decomposition(
    matrix<double>& output,
    const size_t    outer_dim,
    const size_t    dim,
    const size_t    inner_dim,
    const bool      parallelize = false);

MATH_API void decomposition_aad(
    matrix<double>&       output_aad,
    const matrix<double>& output,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize = false);

MATH_API void solve_decomposed(
    vector<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize = false);

MATH_API void solve_decomposed_vectorised(
    matrix<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize = false);

MATH_API void solve_decomposed_aad(
    matrix<double>&       decomposed_aad,
    vector<double>&       x_aad,
    vector<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize = false);

MATH_API void solve_decomposed_aad(
    vector<double>&       x_aad,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim);

MATH_API void multiply(
    vector<double>&       result,
    const vector<double>& in,
    const matrix<double>& mat,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    double                time_multiplier,
    const bool            parallelize = false);

MATH_API void multiply_aad(
    matrix<double>&       mat_aad,
    vector<double>&       in_aad,
    const vector<double>& result_aad,
    const vector<double>& in,
    const matrix<double>& mat,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    double                time_multiplier,
    const bool            parallelize = false);

template <typename T, class Allocator = allocator<T>>
void solve_tridiagonal(T* output, const size_t n, T const* L, T const* D, T* U)
{
    QUARISMA_CHECK_DEBUG(n > 2, "tridiagonal root_finding_algorithms dimension is lower than 3!");

    *U /= *D;
    *output /= *D;

    size_t i = 0;

    for (; i < n - 2; ++i, ++U, ++D, ++L, ++output)
    {
        const T m     = (*(D + 1) - *L * *U);
        *(output + 1) = (*(output + 1) - *L * *output) / m;
        *(U + 1) /= m;
    }
    {
        const T m     = (*(D + 1) - *L * *U);
        *(output + 1) = (*(output + 1) - *L * *output) / m;
    }

    for (auto k = static_cast<int>(n - 2); k >= 0; --k, --U, --output)
    {
        *output -= *U * *(output + 1);
    }
};

};  // namespace tridiagonal_operations
}  // namespace quarisma

#endif