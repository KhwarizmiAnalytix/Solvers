#include "copula/plackett_copula.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "util/exception.h"

namespace quarisma
{

plackett_copula::plackett_copula(double theta) : theta_(theta)
{
    QUARISMA_CHECK(theta > 0.0, "Plackett copula parameter theta must be > 0");
}

plackett_copula::~plackett_copula() = default;

double plackett_copula::evaluate(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        if (std::abs(theta_ - 1.0) < 1e-10)
        {
            return u1 * u2;  // Independence copula
        }

        double sum     = u1 + u2;
        double product = u1 * u2;

        double term1 = 1.0 + (theta_ - 1.0) * sum;
        double term2 = std::sqrt(std::pow(term1, 2.0) - 4.0 * theta_ * (theta_ - 1.0) * product);

        return (term1 - term2) / (2.0 * (theta_ - 1.0));
    }

    // For multivariate case, we only implement the bivariate case
    throw std::runtime_error("Plackett copula is only implemented for bivariate case");
}

double plackett_copula::density(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        if (std::abs(theta_ - 1.0) < 1e-10)
        {
            return 1.0;  // Independence copula
        }

        double sum     = u1 + u2;
        double product = u1 * u2;

        double term1 = theta_ * (1.0 + (theta_ - 1.0) * (sum - 2.0 * product));
        double term2 =
            std::pow(1.0 + (theta_ - 1.0) * sum, 2.0) - 4.0 * theta_ * (theta_ - 1.0) * product;

        return term1 / std::pow(term2, 1.5);
    }

    // For multivariate case, we only implement the bivariate case
    throw std::runtime_error("Plackett copula density is only implemented for bivariate case");
}

size_t plackett_copula::dimension() const
{
    return 2;  // Plackett copula is bivariate
}

void plackett_copula::random_sample(
    double* output, size_t n, uniforms_functoin_type uniforms_func, size_t skip_count) const
{
    // Generate all uniform random numbers at once for better performance
    std::vector<double> uniforms(n * 2);

    // Call the uniforms function to generate random numbers
    uniforms_func(uniforms.data(), n * 2, skip_count);

    // Precompute constants for efficiency
    const double theta_minus_1  = theta_ - 1.0;
    const bool   theta_near_one = std::abs(theta_minus_1) < 1e-10;

    // Transform the uniform random numbers to follow the Plackett copula distribution
    for (size_t i = 0; i < n; ++i)
    {
        const double u1 = uniforms[i * 2];
        const double u2 = uniforms[i * 2 + 1];

        // First marginal is just the uniform
        output[i * 2] = u1;

        // For the second marginal, we need to apply the conditional distribution
        if (theta_near_one)
        {
            // For theta close to 1, use the independence copula
            output[i * 2 + 1] = u2;
        }
        else
        {
            // Compute conditional distribution function
            const double a            = theta_ * u1;
            const double b            = theta_minus_1;
            const double c            = 1.0 + b * u1;
            const double two_a        = 2.0 * a;
            const double one_minus_u2 = 1.0 - u2;

            // Compute terms for the quadratic formula
            const double alpha        = two_a * u2 + c * one_minus_u2;
            const double discriminant = alpha * alpha - 4.0 * a * b * u1 * u2;
            const double beta         = std::sqrt(discriminant);
            const double denominator  = 2.0 * b * u1;

            // Solve for v2 using the conditional distribution
            double v2 = (alpha - beta) / denominator;

            // Ensure v2 is in [0, 1]
            output[i * 2 + 1] = std::max(0.0, std::min(1.0, v2));
        }
    }
}

}  // namespace quarisma
