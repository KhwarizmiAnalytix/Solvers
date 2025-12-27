#include "copula/galambos_copula.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "util/exception.h"

namespace quarisma
{

galambos_copula::galambos_copula(double theta) : theta_(theta)
{
    QUARISMA_CHECK(theta >= 0.0, "Galambos copula parameter theta must be >= 0");
}

galambos_copula::~galambos_copula() = default;

double galambos_copula::evaluate(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        if (u1 <= 0.0 || u2 <= 0.0)
        {
            return 0.0;  // Handle edge case
        }

        if (theta_ == 0.0)
        {
            return u1 * u2;  // Independence copula
        }

        double log_u1     = -std::log(u1);
        double log_u2     = -std::log(u2);
        double pow_log_u1 = std::pow(log_u1, -theta_);
        double pow_log_u2 = std::pow(log_u2, -theta_);
        double sum        = pow_log_u1 + pow_log_u2;

        return u1 * u2 * std::exp(std::pow(sum, -1.0 / theta_));
    }

    // For multivariate case, we only implement the bivariate case
    throw std::runtime_error("Galambos copula is only implemented for bivariate case");
}

double galambos_copula::density(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        if (u1 <= 0.0 || u2 <= 0.0)
        {
            return 0.0;  // Handle edge case
        }

        if (theta_ == 0.0)
        {
            return 1.0;  // Independence copula
        }

        double log_u1     = -std::log(u1);
        double log_u2     = -std::log(u2);
        double pow_log_u1 = std::pow(log_u1, -theta_);
        double pow_log_u2 = std::pow(log_u2, -theta_);
        double sum        = pow_log_u1 + pow_log_u2;
        double pow_sum    = std::pow(sum, -1.0 / theta_);

        // Compute the copula value
        double copula_value = u1 * u2 * std::exp(pow_sum);

        // Compute the density
        double term1 = 1.0 + (1.0 + 1.0 / theta_) * pow_sum;
        double term2 =
            (1.0 + theta_) * pow_log_u1 * pow_log_u2 * std::pow(sum, -2.0 - 1.0 / theta_);

        return copula_value * term1 * term2;
    }

    // For multivariate case, we only implement the bivariate case
    throw std::runtime_error("Galambos copula density is only implemented for bivariate case");
}

size_t galambos_copula::dimension() const
{
    return 2;  // Galambos copula is bivariate
}

double galambos_copula::theta() const
{
    return theta_;
}

void galambos_copula::random_sample(
    double* output, size_t n, uniforms_functoin_type uniforms_func, size_t skip_count) const
{
    // Generate all uniform random numbers at once for better performance
    std::vector<double> uniforms(n * 2);

    // Call the uniforms function to generate random numbers
    uniforms_func(uniforms.data(), n * 2, skip_count);

    // Precompute constants for efficiency
    const double inv_theta        = 1.0 / theta_;
    const double inv_theta_plus_1 = inv_theta + 1.0;
    const double two_pi           = 2.0 * constants::PI;
    const double inv_two_pi       = 1.0 / two_pi;

    // Transform the uniform random numbers to follow the Galambos copula distribution
    for (size_t i = 0; i < n; ++i)
    {
        const double u1 = uniforms[i * 2];
        const double u2 = uniforms[i * 2 + 1];

        // Generate exponential random variable
        const double e = -std::log(u1);

        // Generate uniform random variable on [0, 2*pi]
        const double w = two_pi * u2;

        // Compute the derivative of A at w/(2*pi)
        const double t                 = w * inv_two_pi;
        const double t_theta           = std::pow(t, -theta_);
        const double one_minus_t_theta = std::pow(1.0 - t, -theta_);
        const double sum_pow           = std::pow(t_theta + one_minus_t_theta, -inv_theta_plus_1);
        const double t_theta_plus_1    = std::pow(t, -theta_ - 1.0);

        const double a_prime = -inv_theta * sum_pow * (-theta_ * t_theta_plus_1);

        // Compute common terms
        const double one_minus_t       = 1.0 - t;
        const double one_minus_a_prime = 1.0 - a_prime;
        const double one_plus_a_prime  = 1.0 + a_prime;

        // Transform to Galambos copula
        output[i * 2]     = std::exp(-e * one_minus_t * one_minus_a_prime);
        output[i * 2 + 1] = std::exp(-e * t * one_plus_a_prime);
    }
}

}  // namespace quarisma
