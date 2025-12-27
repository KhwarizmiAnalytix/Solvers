#pragma once

#include <vector>

#include "common/normal_cdf.h"
#include "common/wrapping_hints.h"
#include "copula/copula.h"
#include "matrix_operation/cholesky_decomposition.h"

namespace quarisma
{
class MATH_VISIBILITY gaussian_copula : public copula
{
public:
    /**
     * @brief Constructor for bivariate Gaussian copula
     * @param rho Correlation parameter in [-1, 1]
     */
    MATH_API explicit gaussian_copula(double rho);

    /**
     * @brief Constructor for multivariate Gaussian copula
     * @param correlation_matrix Correlation matrix stored as a flat array (row-major order)
     * @param dimension Dimension of the correlation matrix
     */
    MATH_API gaussian_copula(const double* correlation_matrix, size_t dimension);

    /**
     * @brief Destructor
     */
    MATH_API ~gaussian_copula() override;

    /**
     * @brief Evaluate the Gaussian copula function
     * @param u Vector of uniform variates in [0,1]
     * @return Value of the copula function
     */
    MATH_API double evaluate(const std::vector<double>& u) const override;

    /**
     * @brief Evaluate the Gaussian copula density function
     * @param u Vector of uniform variates in [0,1]
     * @return Value of the copula density function
     */
    MATH_API double density(const std::vector<double>& u) const override;

    /**
     * @brief Generate multiple random samples from the Gaussian copula using a uniform random number generator function
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
     * @brief Get the correlation parameter (for bivariate case)
     * @return The correlation parameter
     */
    MATH_API double rho() const;

private:
    size_t  dim_;                 // Dimension of the correlation matrix
    double* correlation_matrix_;  // Correlation matrix stored as a flat array
    double* cholesky_matrix_;     // Cholesky decomposition of the correlation matrix

    // Helper method to access correlation matrix elements
    double get_correlation(size_t i, size_t j) const;

    // Helper method to access Cholesky matrix elements
    double get_cholesky(size_t i, size_t j) const;
};

}  // namespace quarisma
