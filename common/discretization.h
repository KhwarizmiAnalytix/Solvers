#pragma once

#ifndef __QUARISMA_WRAP__

#include <cstddef>
#include <vector>

#include "MathModule.h"

namespace quarisma
{
namespace discretization
{

/**
 * @brief Generates a uniform discretization of a range with a specified center value.
 * 
 * This function creates a uniform discretization of the range [value_min, value_max],
 * ensuring that the specified 'value' is included in the discretization.
 * 
 * @param[out] output Pointer to the array where the discretized values will be stored.
 * @param[in] n Number of points in the discretization.
 * @param[in] value_min Minimum value of the range.
 * @param[in] value_max Maximum value of the range.
 * @param[in] value Specific value to be included in the discretization.
 */
MATH_API void uniform(
    double* output,
    size_t  n,
    double  value_min,
    double  value_max,
    double  value,
    int     j0     = -1,
    bool    center = false);

/**
 * @brief Generates a non-uniform discretization of a range using an inverse hyperbolic sine transformation.
 * 
 * This function creates a discretization of the range [value_min, value_max] using an inverse
 * hyperbolic sine (asinh) transformation. This results in a denser discretization near the
 * specified 'value' and sparser discretization towards the extremes.
 * 
 * @param[out] output Pointer to the array where the discretized values will be stored.
 * @param[in] n Number of points in the discretization.
 * @param[in] value_min Minimum value of the range.
 * @param[in] value_max Maximum value of the range.
 * @param[in] value Center value for the asinh transformation.
 * @param[in] scaling Scaling factor for the asinh transformation (default is 1.0).
 */
MATH_API void asinh(
    double* output,
    size_t  n,
    double  value_min,
    double  value_max,
    double  value,
    double  scaling = 1.);

/**
 * @brief Extends a vector by uniformly inserting additional points between existing points.
 * 
 * This function takes an input vector and extends it to a specified size by uniformly
 * inserting additional points between the existing points.
 * 
 * @param[in] input Pointer to the array of input values.
 * @param[in] n Size of the input array.
 * @param[in] m Desired size of the extended vector (must be greater than n).
 * @return std::vector<double> Extended vector with uniformly inserted points.
 * @throws std::invalid_argument if m is not greater than n.
 */
MATH_API void extend_vector_uniform(double* output, const double* input, size_t n, size_t m);

}  // namespace discretization
}  // namespace quarisma

#endif