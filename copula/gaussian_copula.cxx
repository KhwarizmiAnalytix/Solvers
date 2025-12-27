#include "copula/gaussian_copula.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "common/constants.h"
#include "common/normal_cdf.h"
#include "distribution/normal_distribution.h"
#include "matrix_operation/linear_solver.h"
#include "util/exception.h"

namespace quarisma
{

// Helper method to access correlation matrix elements
double gaussian_copula::get_correlation(size_t i, size_t j) const
{
    return correlation_matrix_[i * dim_ + j];
}

// Helper method to access Cholesky matrix elements
double gaussian_copula::get_cholesky(size_t i, size_t j) const
{
    return cholesky_matrix_[i * dim_ + j];
}

gaussian_copula::gaussian_copula(double rho) : dim_(2)
{
    QUARISMA_CHECK(rho >= -1.0 && rho <= 1.0, "Correlation parameter must be in [-1, 1]");

    // Allocate memory for the correlation matrix
    correlation_matrix_ = new double[dim_ * dim_];

    // Create a 2x2 correlation matrix
    correlation_matrix_[0] = 1.0;
    correlation_matrix_[1] = rho;
    correlation_matrix_[2] = rho;
    correlation_matrix_[3] = 1.0;

    // Allocate memory for the Cholesky decomposition
    cholesky_matrix_ = new double[dim_ * dim_];

    // Copy the correlation matrix to the Cholesky matrix
    for (size_t i = 0; i < dim_ * dim_; ++i)
    {
        cholesky_matrix_[i] = correlation_matrix_[i];
    }

    // Compute Cholesky decomposition
    bool success = cholesky_decomposition(
        cholesky_matrix_,
        static_cast<quarisma_int>(dim_),
        cholesky_decomposition_enum::LOWER_TRIANGULAR);

    QUARISMA_CHECK(success, "Correlation matrix is not positive definite");
}

gaussian_copula::gaussian_copula(const double* corr_matrix, size_t dim) : dim_(dim)
{
    // Allocate memory for the correlation matrix
    correlation_matrix_ = new double[dim_ * dim_];

    // Copy the input correlation matrix
    for (size_t i = 0; i < dim_ * dim_; ++i)
    {
        correlation_matrix_[i] = corr_matrix[i];
    }

    // Check if the matrix is square (already ensured by the constructor)

    // Check if the diagonal elements are 1
    for (size_t i = 0; i < dim_; ++i)
    {
        QUARISMA_CHECK(
            std::abs(get_correlation(i, i) - 1.0) <= 1e-10,
            "Diagonal elements of correlation matrix must be 1");
    }

    // Allocate memory for the Cholesky decomposition
    cholesky_matrix_ = new double[dim_ * dim_];

    // Copy the correlation matrix to the Cholesky matrix
    for (size_t i = 0; i < dim_ * dim_; ++i)
    {
        cholesky_matrix_[i] = correlation_matrix_[i];
    }

    // Compute Cholesky decomposition
    bool success = cholesky_decomposition(
        cholesky_matrix_,
        static_cast<quarisma_int>(dim_),
        cholesky_decomposition_enum::LOWER_TRIANGULAR);

    QUARISMA_CHECK(success, "Correlation matrix is not positive definite");
}

gaussian_copula::~gaussian_copula()
{
    delete[] correlation_matrix_;
    delete[] cholesky_matrix_;
}

double gaussian_copula::evaluate(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    QUARISMA_CHECK(u.size() == dim_, "Input dimension does not match correlation matrix dimension");

    // For bivariate case, we can use a more efficient formula
    if (u.size() == 2)
    {
        double rho = get_correlation(0, 1);
        double x   = inv_normalcdf(u[0]);
        double y   = inv_normalcdf(u[1]);

        // Bivariate normal CDF
        double h = (x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho));
        double a = std::sqrt(1 - rho * rho);

        return (1.0 / (2.0 * constants::PI * a)) * std::exp(-h);
    }

    // For higher dimensions, we need to use numerical methods
    // This is a simplified implementation that doesn't handle the general case well
    // A more robust implementation would use numerical integration
    throw std::runtime_error("Evaluation of Gaussian copula for dimensions > 2 is not implemented");
}

double gaussian_copula::density(const std::vector<double>& u) const
{
    QUARISMA_CHECK(
        std::all_of(u.begin(), u.end(), [](double val) { return val >= 0.0 && val <= 1.0; }),
        "Input values must be in [0,1]");

    QUARISMA_CHECK(u.size() == dim_, "Input dimension does not match correlation matrix dimension");

    // Transform u to normal quantiles
    std::vector<double> z(u.size());
    for (size_t i = 0; i < u.size(); ++i)
    {
        z[i] = inv_normalcdf(u[i]);
    }

    // Compute determinant (product of diagonal elements of Cholesky decomposition squared)
    double det = 1.0;
    for (size_t i = 0; i < dim_; ++i)
    {
        det *= get_cholesky(i, i) * get_cholesky(i, i);
    }

    // Compute z^T * (R^-1 - I) * z
    // First compute R^-1 * z
    std::vector<double>     temp_z = z;
    std::vector<quarisma_int> pivot(dim_);

    // Create a copy of the correlation matrix for solving the system
    double* temp_matrix = new double[dim_ * dim_];
    for (size_t i = 0; i < dim_ * dim_; ++i)
    {
        temp_matrix[i] = correlation_matrix_[i];
    }

    // Solve the system R * x = z to get R^-1 * z
    linear_solver(
        temp_matrix,
        pivot.data(),
        static_cast<quarisma_int>(dim_),
        temp_z.data(),
        linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    // Compute z^T * (R^-1 * z - z)
    double exponent = 0.0;
    for (size_t i = 0; i < z.size(); ++i)
    {
        exponent += z[i] * (temp_z[i] - z[i]);
    }
    exponent *= -0.5;

    // Clean up
    delete[] temp_matrix;

    return std::exp(exponent) / std::sqrt(det);
}

size_t gaussian_copula::dimension() const
{
    return dim_;
}

double gaussian_copula::rho() const
{
    if (dim_ == 2)
    {
        return get_correlation(0, 1);
    }
    else
    {
        throw std::runtime_error("rho() is only valid for bivariate copulas");
    }
}

void gaussian_copula::random_sample(
    double* output, size_t n, uniforms_functoin_type uniforms_func, size_t skip_count) const
{
    // Generate all uniform random numbers at once for better performance
    std::vector<double> uniforms(n * dim_);

    // Call the uniforms function to generate random numbers
    uniforms_func(uniforms.data(), n * dim_, skip_count);

    // Temporary storage for normal variates (reused for both z and x)
    std::vector<double> temp(dim_);

    // Transform the uniform random numbers to follow the Gaussian copula distribution
    for (size_t i = 0; i < n; ++i)
    {
        // Extract the uniform random numbers for this sample and transform to standard normal
        for (size_t j = 0; j < dim_; ++j)
        {
            temp[j] = inv_normalcdf(uniforms[i * dim_ + j]);
        }

        // Make a copy of the standard normal variates
        std::vector<double> z(temp);

        // Apply Cholesky decomposition to get correlated normal variables
        for (size_t j = 0; j < dim_; ++j)
        {
            temp[j] = 0.0;
            for (size_t k = 0; k <= j; ++k)
            {
                temp[j] += get_cholesky(j, k) * z[k];
            }
        }

        // Transform to uniform using the normal CDF and store in output
        for (size_t j = 0; j < dim_; ++j)
        {
            output[i * dim_ + j] = normalcdf(temp[j]);
        }
    }
}

}  // namespace quarisma
