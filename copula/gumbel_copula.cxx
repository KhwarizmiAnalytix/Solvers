#include "copula/gumbel_copula.h"

#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "common/constants.h"
#include "util/exception.h"

namespace quarisma
{
gumbel_copula::gumbel_copula(double theta) : theta_(theta)
{
    QUARISMA_CHECK(theta >= 1.0, "Gumbel copula parameter theta must be >= 1");
}

double gumbel_copula::evaluate(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // Sum of (-ln(u_i))^theta
    double sum = 0.0;
    for (const double& ui : u)
    {
        if (ui <= 0.0)
        {
            return 0.0;  // Handle edge case
        }
        sum += std::pow(-std::log(ui), theta_);
    }

    return std::exp(-std::pow(sum, 1.0 / theta_));
}

double gumbel_copula::density(const std::vector<double>& u) const
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

        double log_u1     = -std::log(u1);
        double log_u2     = -std::log(u2);
        double pow_log_u1 = std::pow(log_u1, theta_);
        double pow_log_u2 = std::pow(log_u2, theta_);
        double sum        = pow_log_u1 + pow_log_u2;
        double pow_sum    = std::pow(sum, 1.0 / theta_);

        double term1 = std::exp(-pow_sum);
        double term2 = std::pow(log_u1 * log_u2, theta_ - 1.0);
        double term3 =
            (1.0 + (theta_ - 1.0) * std::pow(pow_log_u1 * pow_log_u2 / sum, 1.0 / theta_));
        double term4 = std::pow(u1 * u2 * sum, -1.0);

        return term1 * term2 * term3 * term4;
    }

    // For multivariate case (d-dimensional)
    int d = (int)u.size();

    // Check for edge cases
    for (const double& ui : u)
    {
        if (ui <= 0.0)
        {
            return 0.0;
        }
    }

    // Calculate vector of -ln(u_i)
    std::vector<double> log_u(d);
    for (size_t i = 0; i < d; ++i)
    {
        log_u[i] = -std::log(u[i]);
    }

    // Calculate vector of (-ln(u_i))^theta
    std::vector<double> pow_log_u(d);
    for (size_t i = 0; i < d; ++i)
    {
        pow_log_u[i] = std::pow(log_u[i], theta_);
    }

    // Calculate sum of (-ln(u_i))^theta
    double sum = std::accumulate(pow_log_u.begin(), pow_log_u.end(), 0.0);

    // Calculate the density (this is a simplified approximation)
    double pow_sum  = std::pow(sum, 1.0 / theta_);
    double exp_term = std::exp(-pow_sum);

    double product_term = 1.0;
    for (size_t i = 0; i < d; ++i)
    {
        product_term *= std::pow(log_u[i], theta_ - 1.0) / u[i];
    }

    // This is a simplified approximation of the density
    return exp_term * product_term * std::pow(sum, -d + 1.0 / theta_);
}

size_t gumbel_copula::dimension() const
{
    return 2;  // Gumbel copula is bivariate
}

double gumbel_copula::theta() const
{
    return theta_;
}

void gumbel_copula::random_sample(
    double* output, size_t n, uniforms_functoin_type uniforms_func, size_t skip_count) const
{
    // For Gumbel copula, we only need n uniform random numbers (not 2*n)
    // since we use a special sampling method based on stable distributions
    std::vector<double> uniforms(n * 2);

    // Call the uniforms function to generate random numbers
    // We still generate 2*n uniforms to maintain the same interface as other copulas,
    // but we'll only use the first uniform from each pair
    uniforms_func(uniforms.data(), n * 2, skip_count);

    // Precompute constants for efficiency
    const double inv_theta = 1.0 / theta_;
    const double angle     = constants::PI / (2.0 * theta_);
    const double cos_angle = std::cos(angle);
    const double sin_angle = std::sin(angle);

    // Transform the uniform random numbers to follow the Gumbel copula distribution
    // Note: For the Gumbel copula, we only need one uniform random variable per dimension
    // The first uniform is used to generate the stable random variable via inverse transform
    // The second dimension is derived from the first using the angle transformation
    for (size_t i = 0; i < n; ++i)
    {
        // We only need the first uniform for each pair
        const double u = uniforms[i * 2];

        // Generate stable random variable using inverse transform sampling
        const double gamma = -std::log(u);

        // Precompute common terms - we use trigonometric transformation to generate
        // the second dimension rather than using the second uniform random variable
        const double gamma_cos = gamma * cos_angle;
        const double gamma_sin = gamma * sin_angle;

        // Transform to Gumbel copula using the Laplace transform method
        output[i * 2]     = std::exp(-std::pow(gamma_cos, inv_theta));
        output[i * 2 + 1] = std::exp(-std::pow(gamma_sin, inv_theta));
    }
}

}  // namespace quarisma
