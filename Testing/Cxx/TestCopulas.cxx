#include <cmath>
#include <vector>

#include "common/constants.h"
#include "common/pointer.h"
#include "copula/clayton_copula.h"
#include "copula/copula.h"
#include "copula/frank_copula.h"
#include "copula/galambos_copula.h"
#include "copula/gaussian_copula.h"
#include "copula/gumbel_copula.h"
#include "copula/plackett_copula.h"
#include "util/logger.h"
#include "well/well_19937ac.h"
#include "quarismaTest.h"

using namespace quarisma;

namespace
{
// Helper function to generate uniform random numbers using well_19937ac
void generate_uniforms(double* out, size_t count, size_t skip)
{
    static well_19937ac generator(12345, 0);
    generator.uniforms(out, count, skip);
}

// Test the Gaussian copula
void test_gaussian_copula()
{
    // Create a Gaussian copula with correlation 0.5
    gaussian_copula copula(0.5);

    // Test the dimension
    EXPECT_EQ(copula.dimension(), 2);

    // Test the correlation parameter
    EXPECT_DOUBLE_EQ(copula.rho(), 0.5);

    // Test the copula function
    std::vector<double> u     = {0.5, 0.5};
    double              value = copula.evaluate(u);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    // Test the density function
    double density = copula.density(u);
    EXPECT_GT(density, 0.0);

    // Test with a correlation matrix
    double          corr_data[4] = {1.0, 0.7, 0.7, 1.0};
    gaussian_copula copula2(corr_data, 2);
    EXPECT_EQ(copula2.dimension(), 2);
    EXPECT_DOUBLE_EQ(copula2.rho(), 0.7);

    // Test sampling with a custom random generator
    QUARISMA_LOGF(INFO, "Testing Gaussian copula with custom random generator");

    // Test multiple samples with generator using flat memory
    size_t              n   = 10;
    size_t              dim = copula.dimension();
    std::vector<double> samples(n * dim);

    copula.random_sample(samples.data(), n, generate_uniforms);

    // Verify the samples
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            EXPECT_GT(samples[i * dim + j], 0.0);
            EXPECT_LE(samples[i * dim + j], 1.0);
        }
    }

    // Test with skip count
    std::vector<double> samples_discard(n * dim);
    copula.random_sample(samples_discard.data(), n, generate_uniforms, 100);

    // Test error handling
    ASSERT_ANY_THROW({
        gaussian_copula invalid_copula(2.0);  // Invalid correlation parameter
    });

    // Test with invalid inputs
    std::vector<double> invalid_u = {-0.1, 0.5};
    ASSERT_ANY_THROW({ copula.evaluate(invalid_u); });

    std::vector<double> invalid_dim = {0.5, 0.5, 0.5};
    ASSERT_ANY_THROW({ copula.evaluate(invalid_dim); });
}

// Test the Clayton copula
void test_clayton_copula()
{
    // Create a Clayton copula with parameter 2.0
    clayton_copula copula(2.0);

    // Test the parameter
    EXPECT_DOUBLE_EQ(copula.theta(), 2.0);

    // Test the dimension
    EXPECT_EQ(copula.dimension(), 2);

    // Test the copula function
    std::vector<double> u     = {0.5, 0.5};
    double              value = copula.evaluate(u);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    // Test the density function
    double density = copula.density(u);
    EXPECT_GT(density, 0.0);

    // Test sampling with a custom random generator
    QUARISMA_LOGF(INFO, "Testing Clayton copula with custom random generator");

    // Test multiple samples with generator using flat memory
    size_t              n   = 10;
    size_t              dim = copula.dimension();
    std::vector<double> samples(n * dim);

    copula.random_sample(samples.data(), n, generate_uniforms);

    // Verify the samples
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            EXPECT_GT(samples[i * dim + j], 0.0);
            EXPECT_LE(samples[i * dim + j], 1.0);
        }
    }

    // Test error handling
    ASSERT_ANY_THROW({
        clayton_copula invalid_copula(-1.0);  // Invalid theta parameter
    });
}

// Test the Frank copula
void test_frank_copula()
{
    // Create a Frank copula with parameter 5.0
    frank_copula copula(5.0);

    // Test the parameter
    EXPECT_DOUBLE_EQ(copula.theta(), 5.0);

    // Test the dimension
    EXPECT_EQ(copula.dimension(), 2);

    // Test the copula function
    std::vector<double> u     = {0.5, 0.5};
    double              value = copula.evaluate(u);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    // Test the density function
    double density = copula.density(u);
    EXPECT_GT(density, 0.0);

    // Test sampling with a custom random generator
    QUARISMA_LOGF(INFO, "Testing Frank copula with custom random generator");

    // Test multiple samples with generator using flat memory
    size_t              n   = 10;
    size_t              dim = copula.dimension();
    std::vector<double> samples(n * dim);

    copula.random_sample(samples.data(), n, generate_uniforms);

    // Verify the samples
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            EXPECT_GT(samples[i * dim + j], 0.0);
            EXPECT_LE(samples[i * dim + j], 1.0);
        }
    }

    // Test error handling
    ASSERT_ANY_THROW({
        frank_copula invalid_copula(0.0);  // Invalid theta parameter
    });
}

// Test the Gumbel copula
void test_gumbel_copula()
{
    // Create a Gumbel copula with parameter 2.0
    gumbel_copula copula(2.0);

    // Test the parameter
    EXPECT_DOUBLE_EQ(copula.theta(), 2.0);

    // Test the dimension
    EXPECT_EQ(copula.dimension(), 2);

    // Test the copula function
    std::vector<double> u     = {0.5, 0.5};
    double              value = copula.evaluate(u);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    // Test the density function
    double density = copula.density(u);
    EXPECT_GT(density, 0.0);

    // Test sampling with a custom random generator
    QUARISMA_LOGF(INFO, "Testing Gumbel copula with custom random generator");

    // Test multiple samples with generator using flat memory
    size_t              n   = 10;
    size_t              dim = copula.dimension();
    std::vector<double> samples(n * dim);

    copula.random_sample(samples.data(), n, generate_uniforms);

    // Verify the samples
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            EXPECT_GT(samples[i * dim + j], 0.0);
            EXPECT_LE(samples[i * dim + j], 1.0);
        }
    }

    // Test error handling
    ASSERT_ANY_THROW({
        gumbel_copula invalid_copula(0.5);  // Invalid theta parameter
    });
}

// Test the Galambos copula
void test_galambos_copula()
{
    // Create a Galambos copula with parameter 1.0
    galambos_copula copula(1.0);

    // Test the parameter
    EXPECT_DOUBLE_EQ(copula.theta(), 1.0);

    // Test the dimension
    EXPECT_EQ(copula.dimension(), 2);

    // Test the copula function
    std::vector<double> u     = {0.5, 0.5};
    double              value = copula.evaluate(u);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    // Test the density function
    double density = copula.density(u);
    EXPECT_GT(density, 0.0);

    // Test sampling with a custom random generator
    QUARISMA_LOGF(INFO, "Testing Galambos copula with custom random generator");

    // Test multiple samples with generator using flat memory
    size_t              n   = 10;
    size_t              dim = copula.dimension();
    std::vector<double> samples(n * dim);

    copula.random_sample(samples.data(), n, generate_uniforms);

    // Verify the samples
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            EXPECT_GT(samples[i * dim + j], 0.0);
            EXPECT_LE(samples[i * dim + j], 1.0);
        }
    }

    // Test error handling
    ASSERT_ANY_THROW({
        galambos_copula invalid_copula(-0.5);  // Invalid theta parameter
    });
}

// Test the Plackett copula
void test_plackett_copula()
{
    // Create a Plackett copula with parameter 2.0
    plackett_copula copula(2.0);

    // Test the parameter
    EXPECT_DOUBLE_EQ(copula.theta(), 2.0);

    // Test the dimension
    EXPECT_EQ(copula.dimension(), 2);

    // Test the copula function
    std::vector<double> u     = {0.5, 0.5};
    double              value = copula.evaluate(u);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    // Test the density function
    double density = copula.density(u);
    EXPECT_GT(density, 0.0);

    // Test sampling with a custom random generator
    QUARISMA_LOGF(INFO, "Testing Plackett copula with custom random generator");

    // Test multiple samples with generator using flat memory
    size_t              n   = 10;
    size_t              dim = copula.dimension();
    std::vector<double> samples(n * dim);

    copula.random_sample(samples.data(), n, generate_uniforms);

    // Verify the samples
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            EXPECT_GT(samples[i * dim + j], 0.0);
            EXPECT_LE(samples[i * dim + j], 1.0);
        }
    }

    // Test error handling
    ASSERT_ANY_THROW({
        plackett_copula invalid_copula(0.0);  // Invalid theta parameter
    });
}

// Test the base Copula class functionality
void test_base_copula()
{
    // Create a concrete copula instance
    clayton_copula  clayton_cop(2.0);
    frank_copula    frank_cop(5.0);
    gumbel_copula   gumbel_cop(2.0);
    galambos_copula galambos_cop(1.0);
    plackett_copula plackett_cop(2.0);

    // Create a pointer to the base class
    ptr_const<copula> copulas[] = {
        quarisma::util::make_ptr_const<gaussian_copula>(0.5),
        quarisma::util::make_ptr_const<clayton_copula>(clayton_cop),
        quarisma::util::make_ptr_const<frank_copula>(frank_cop),
        quarisma::util::make_ptr_const<gumbel_copula>(2.),
        quarisma::util::make_ptr_const<galambos_copula>(galambos_cop),
        quarisma::util::make_ptr_const<plackett_copula>(plackett_cop)};

    // Test the virtual methods through the base class pointer
    for (const auto& cop : copulas)
    {
        // Test the dimension method
        EXPECT_EQ(cop->dimension(), 2);

        // Test the random_sample method with a custom random generator
        size_t              n   = 5;
        size_t              dim = cop->dimension();
        std::vector<double> samples(n * dim);

        // Call the virtual method through the base class pointer
        cop->random_sample(samples.data(), n, generate_uniforms);

        // Verify the samples
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < dim; ++j)
            {
                EXPECT_GT(samples[i * dim + j], 0.0);
                EXPECT_LE(samples[i * dim + j], 1.0);
            }
        }
    }
}
}  // namespace

QUARISMATEST(Math, Copulas)
{
    START_LOG_TO_FILE_NAME(Copulas);

    QUARISMA_LOGF(INFO, "Testing Gaussian copula");
    test_gaussian_copula();

    QUARISMA_LOGF(INFO, "Testing Clayton copula");
    test_clayton_copula();

    QUARISMA_LOGF(INFO, "Testing Frank copula");
    test_frank_copula();

    QUARISMA_LOGF(INFO, "Testing Gumbel copula");
    test_gumbel_copula();

    QUARISMA_LOGF(INFO, "Testing Galambos copula");
    test_galambos_copula();

    QUARISMA_LOGF(INFO, "Testing Plackett copula");
    test_plackett_copula();

    QUARISMA_LOGF(INFO, "Testing base copula functionality");
    test_base_copula();

    END_LOG_TO_FILE_NAME(Copulas);
    END_TEST();
}
