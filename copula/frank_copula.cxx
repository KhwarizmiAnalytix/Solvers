#include "copula/frank_copula.h"

#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "util/exception.h"

namespace quarisma
{

frank_copula::frank_copula(double theta) : theta_(theta)
{
    QUARISMA_CHECK(theta != 0.0, "Frank copula parameter theta must not be zero");
}

frank_copula::~frank_copula() = default;

double frank_copula::evaluate(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        double u1 = u[0];
        double u2 = u[1];

        double num = (std::exp(-theta_ * u1) - 1.0) * (std::exp(-theta_ * u2) - 1.0);
        double den = std::exp(-theta_) - 1.0;

        return -1.0 / theta_ * std::log(1.0 + num / den);
    }

    // For multivariate case
    double exp_theta = std::exp(-theta_);
    double den       = exp_theta - 1.0;

    double product = 1.0;
    for (const double& ui : u)
    {
        product *= (std::exp(-theta_ * ui) - 1.0) / den;
    }

    return -1.0 / theta_ * std::log(1.0 + product * den);
}

//tex: $$\frac{\theta(1-e^{-\theta})e^{-\theta(u+v)}}{[(1-e^{-\theta}) - (1-e^{-\theta u})(1-e^{-\theta v})]^2}$$
//$$C(u_1, u_2, ..., u_n; \theta) = -\frac{1}{\theta} \ln \left(1 + \frac{\prod_{i=1}^{n} (e^{-\theta u_i} - 1)}{(e^{-\theta} - 1)^{n-1}}\right)$$

double frank_copula::density(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    // For bivariate case
    if (u.size() == 2)
    {
        const auto u1 = u[0];
        const auto u2 = u[1];

        const auto exp_theta    = std::exp(-theta_);
        const auto exp_theta_u1 = std::exp(-theta_ * u1);
        const auto exp_theta_u2 = std::exp(-theta_ * u2);

        const auto num = theta_ * (1. - exp_theta) * exp_theta_u2 * exp_theta_u1;
        const auto den =
            std::pow((exp_theta - 1.0) + (exp_theta_u1 - 1.0) * (exp_theta_u2 - 1.0), 2.0);

        return num / den;
    }

    // For multivariate case (d-dimensional)
    int    d         = (int)u.size();
    double exp_theta = std::exp(-theta_);
    double den       = exp_theta - 1.0;

    // Calculate product of (exp(-theta*u_i) - 1)
    double product = 1.0;
    for (const double& ui : u)
    {
        product *= (std::exp(-theta_ * ui) - 1.0);
    }

    // Calculate sum of exp(-theta*u_i)
    double sum_exp = 0.0;
    for (const double& ui : u)
    {
        sum_exp += std::exp(-theta_ * ui);
    }

    // Calculate the density
    double term1 =
        std::pow(theta_, d - 1) * std::exp(theta_ * std::accumulate(u.begin(), u.end(), 0.0));
    double term2 = product;
    double term3 = std::pow(den, 1 - d);
    double term4 = std::pow(1.0 + product / den, -d - 1);

    return term1 * term2 * term3 * term4;
}

size_t frank_copula::dimension() const
{
    return 2;  // Frank copula is bivariate
}

double frank_copula::theta() const
{
    return theta_;
}

void frank_copula::random_sample(
    double* output, size_t n, uniforms_functoin_type uniforms_func, size_t skip_count) const
{
    // Generate all uniform random numbers at once for better performance
    std::vector<double> uniforms(n * 2);

    // Call the uniforms function to generate random numbers
    uniforms_func(uniforms.data(), n * 2, skip_count);

    // Precompute constants for efficiency
    const double inv_theta         = 1.0 / theta_;
    const double exp_theta         = std::exp(-theta_);
    const double exp_theta_minus_1 = exp_theta - 1.0;
    const bool   small_theta       = std::abs(theta_) <= 1e-10;

    // Transform the uniform random numbers to follow the Frank copula distribution
    for (size_t i = 0; i < n; ++i)
    {
        const double u1 = uniforms[i * 2];
        const double u2 = uniforms[i * 2 + 1];

        // First marginal is just the uniform
        output[i * 2] = u1;

        // For the second marginal, we need to apply the conditional distribution
        if (small_theta)
        {
            // For small theta, use the independence copula
            output[i * 2 + 1] = u2;
        }
        else
        {
            const double exp_theta_u1 = std::exp(-theta_ * u1);
            const double numerator    = u2 * exp_theta_minus_1;
            const double denominator  = exp_theta_u1 * (1.0 - u2) + u2;

            // Compute the second marginal using the conditional distribution
            output[i * 2 + 1] = -inv_theta * std::log(1.0 + numerator / denominator);
        }
    }
}

}  // namespace quarisma
