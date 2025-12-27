#pragma once

#include <functional>
#include <vector>

#include "MathModule.h"
#include "common/wrapping_hints.h"

namespace quarisma
{
class copula
{
public:
    using uniforms_functoin_type = std::function<void(double*, size_t, size_t)>;

    /**
     * @brief Virtual destructor
     */
    virtual ~copula() = default;

    /**
     * @brief Evaluate the copula function
     * @param u Vector of uniform variates in [0,1]
     * @return Value of the copula function
     */
    virtual double evaluate(const std::vector<double>& u) const = 0;

    /**
     * @brief Evaluate the copula density function
     * @param u Vector of uniform variates in [0,1]
     * @return Value of the copula density function
     */
    virtual double density(const std::vector<double>& u) const = 0;

    /**
     * @brief Get the dimension of the copula
     * @return The dimension of the copula
     */
    virtual size_t dimension() const = 0;

    /**
     * @brief Generate multiple random samples from the copula using a uniform random number generator function
     * @param output Pointer to pre-allocated memory for the output samples (size: n * dimension())
     * @param n Number of samples to generate
     * @param uniforms_func Function pointer to generate uniform random numbers
     * @param skip_count Number of values to skip before generating samples
     */
    QUARISMA_WRAPEXCLUDE virtual void random_sample(
        double*                output,
        size_t                 n,
        uniforms_functoin_type uniforms_func,
        size_t                 skip_count = 0) const = 0;
};

}  // namespace quarisma
