#include "copula/clayton_copula.h"

#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "util/exception.h"

namespace quarisma
{

clayton_copula::clayton_copula(double theta) : theta_(theta)
{
    QUARISMA_CHECK(theta > 0.0, "Clayton copula parameter theta must be positive");
}

clayton_copula::~clayton_copula() = default;

double clayton_copula::evaluate(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        return std::pow(std::pow(u1, -theta_) + std::pow(u2, -theta_) - 1.0, -1.0 / theta_);
    }

    // For multivariate case
    double sum = 0.0;
    for (const double& ui : u)
    {
        sum += std::pow(ui, -theta_);
    }
    sum -= u.size() - 1.0;

    return std::pow(sum, -1.0 / theta_);
}

double clayton_copula::density(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        double term1 = (1.0 + theta_) * std::pow(u1 * u2, -(1.0 + theta_));
        double term2 =
            std::pow(std::pow(u1, -theta_) + std::pow(u2, -theta_) - 1.0, -2.0 - 1.0 / theta_);

        return term1 * term2;
    }

    // For multivariate case (d-dimensional)
    size_t d = u.size();

    // Product of (theta*k + 1) for k = 0 to d-2
    double product_term = 1.0;
    for (size_t k = 0; k < d - 1; ++k)
    {
        product_term *= (theta_ * k + 1.0);
    }

    // Product of u_i^(-theta-1)
    double product_ui = 1.0;
    for (const double& ui : u)
    {
        product_ui *= std::pow(ui, -theta_ - 1.0);
    }

    // Sum of u_i^(-theta)
    double sum_ui = 0.0;
    for (const double& ui : u)
    {
        sum_ui += std::pow(ui, -theta_);
    }

    // Final term
    double final_term = std::pow(sum_ui - d + 1.0, -1.0 / theta_ - d);

    return product_term * product_ui * final_term;
}

size_t clayton_copula::dimension() const
{
    return 2;  // Clayton copula is bivariate
}

double clayton_copula::theta() const
{
    return theta_;
}

void clayton_copula::random_sample(
    double* output, size_t n, uniforms_functoin_type uniforms_func, size_t skip_count) const
{
    // Generate all uniform random numbers at once for better performance
    std::vector<double> uniforms(n * 2);

    // Call the uniforms function to generate random numbers
    uniforms_func(uniforms.data(), n * 2, skip_count);

    // Precompute constants for efficiency
    const double inv_theta   = 1.0 / theta_;
    const double theta_ratio = -theta_ / (1.0 + theta_);

    // Transform the uniform random numbers to follow the Clayton copula distribution
    for (size_t i = 0; i < n; ++i)
    {
        const double u1 = uniforms[i * 2];
        const double u2 = uniforms[i * 2 + 1];

        // First marginal is just the uniform
        output[i * 2] = u1;

        // For the second marginal, we need to apply the conditional distribution
        const double u1_neg_theta = std::pow(u1, -theta_);
        const double u2_power     = std::pow(u2, theta_ratio);

        // Compute the second marginal using the conditional distribution
        output[i * 2 + 1] = 1. / std::pow(1.0 + u2_power * (u1_neg_theta - 1.0), inv_theta);
    }
}

}  // namespace quarisma
