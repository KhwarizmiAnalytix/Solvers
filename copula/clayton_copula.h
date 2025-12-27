#pragma once

#include <cmath>

#include "common/wrapping_hints.h"
#include "copula/copula.h"

namespace quarisma
{
class MATH_VISIBILITY clayton_copula : public copula
{
public:
    /**
     * @brief Constructor
     * @param theta Parameter controlling the strength of dependence (theta > 0)
     */
    MATH_API explicit clayton_copula(double theta);

    /**
     * @brief Virtual destructor
     */
    MATH_API ~clayton_copula() override;

    /**
     * @brief Evaluate the Clayton copula function
     * @param u Vector of uniform variates in [0,1]
     * @return Value of the copula function
     */
    MATH_API double evaluate(const std::vector<double>& u) const override;

    /**
     * @brief Evaluate the Clayton copula density function
     * @param u Vector of uniform variates in [0,1]
     * @return Value of the copula density function
     */
    MATH_API double density(const std::vector<double>& u) const override;

    /**
     * @brief Generate multiple random samples from the Clayton copula using a uniform random number generator function
     * @param output Pointer to pre-allocated memory for the output samples (size: n * dimension())
     * @param n Number of samples to generate
     * @param uniforms_func Function pointer to generate uniform random numbers
     * @param skip_count Number of values to skip before generating samples
     */
    QUARISMA_WRAPEXCLUDE MATH_API void random_sample(
        double*                output,
        size_t                 n,
        uniforms_functoin_type uniforms_func,
        size_t                 skip_count = 0) const override;

    /**
     * @brief Get the dimension of the copula
     * @return The dimension of the copula
     */
    MATH_API size_t dimension() const override;

    /**
     * @brief Get the parameter theta
     * @return The parameter theta
     */
    MATH_API double theta() const;

private:
    double theta_;
};

}  // namespace quarisma
