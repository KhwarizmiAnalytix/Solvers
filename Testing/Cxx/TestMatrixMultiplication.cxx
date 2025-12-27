#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/constants.h"
#include "common/packet.h"
#include "common/pointer.h"
#include "terminals/matrix.h"
#include "quarismaTest.h"
#include "quarismaTestingHelper.h"

//#define DEBUG_MATRIXMULTIPLICATION

using namespace quarisma;

namespace
{
template <typename>
struct matrixTolerance
{
};

template <>
struct matrixTolerance<float>
{
    static constexpr float tolerance = (float)1.5e-4;
};

template <>
struct matrixTolerance<double>
{
    static constexpr double tolerance = 5.e-13;
};

template <typename value_t>
class temp_class
{
public:
    explicit temp_class(quarisma::matrix<value_t>&& a) : a_(std::move(a)) {};

    const auto& a() const { return a_; }

private:
    quarisma::matrix<value_t> a_;
};

template <typename value_t>
void vector_matrix_multiplication(int rows, int columns)
{
    constexpr auto tol = matrixTolerance<value_t>::tolerance;

    quarisma::matrix<value_t> A(rows, columns);
    quarisma::vector<value_t> v(rows);
    quarisma::vector<value_t> r(columns);

    std::default_random_engine             generator;
    std::uniform_real_distribution<double> distribution(-5., 5.);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            A[i][j] = (value_t)distribution(generator);
        }
    }
    for (int j = 0; j < rows; ++j)
    {
        v[j] = (value_t)distribution(generator);
    }

    r = v * A;

    quarisma::vector<value_t> r_seq(columns);
    for (int j = 0; j < columns; ++j)
    {
        double sum = 0.;
        for (int i = 0; i < rows; ++i)
        {
            sum += v[i] * A[i][j];
        }
        r_seq[j] = (value_t)sum;
    }
    auto max_error = hmax(fabs(r - r_seq));

#ifdef DEBUG_MATRIXMULTIPLICATION
    QUARISMA_LOGF(INFO, " =========== vector matrix multiplication =========");
    QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
    QUARISMA_LOGF(INFO, " ==================================================");
#else
    EXPECT_LT(max_error, tol);
#endif
}

template <typename value_t>
void matrix_vector_multiplication(int rows, int columns)
{
    constexpr auto tol = matrixTolerance<value_t>::tolerance;

    quarisma::matrix<value_t> A(rows, columns);
    quarisma::vector<value_t> v(columns);
    quarisma::vector<value_t> r(rows);

    std::default_random_engine             generator;
    std::uniform_real_distribution<double> distribution(-5., 5.);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            A[i][j] = (value_t)distribution(generator);
        }
    }
    for (int j = 0; j < columns; ++j)
    {
        v[j] = (value_t)distribution(generator);
    }

    r = A * v;

    double max_error = 0;
    for (int i = 0; i < rows; ++i)
    {
        double sum = 0.;
        for (int j = 0; j < columns; ++j)
        {
            sum += A[i][j] * v[j];
        }
        max_error = std::fmax(max_error, std::fabs(sum - r[i]));
    }

#ifdef DEBUG_MATRIXMULTIPLICATION
    QUARISMA_LOGF(INFO, " =========== matrix vector multiplication =========");
    QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
    QUARISMA_LOGF(INFO, " ==================================================");
#else
    EXPECT_LE(max_error, tol);
#endif

    quarisma::matrix<value_t> B(columns, rows);

    B = transpose(A);

    max_error = 0.;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            max_error = std::fmax(max_error, std::fabs(A[i][j] - B[j][i]));
        }
    }
#ifdef DEBUG_MATRIXMULTIPLICATION
    QUARISMA_LOGF(INFO, " =========== matrix transposition =========");
    QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
    QUARISMA_LOGF(INFO, " ==========================================");
#else
    EXPECT_LE(max_error, tol);
#endif
}

template <typename value_t>
void matrix_multiplication(int rows, int columns, int depth, bool transpose_a, bool transpose_b)
{
    constexpr auto tol       = matrixTolerance<value_t>::tolerance;
    auto           a_rows    = rows;
    auto           a_columns = depth;
    if (transpose_a)
    {
        std::swap(a_rows, a_columns);
    }

    auto b_rows    = depth;
    auto b_columns = columns;
    if (transpose_b)
    {
        std::swap(b_rows, b_columns);
    }

    quarisma::matrix<value_t> A(a_rows, a_columns);
    quarisma::matrix<value_t> B(b_rows, b_columns);
    quarisma::matrix<value_t> C(rows, columns);

    std::default_random_engine             generator;
    std::uniform_real_distribution<double> distribution(-5., 5.);

    for (int i = 0; i < a_rows; ++i)
    {
        for (int j = 0; j < a_columns; ++j)
        {
            A[i][j] = (value_t)distribution(generator);
        }
    }

    for (int i = 0; i < b_rows; ++i)
    {
        for (int j = 0; j < b_columns; ++j)
        {
            B[i][j] = (value_t)distribution(generator);
        }
    }

    if (transpose_a && transpose_b)
    {
        C                 = transpose(A) * transpose(B);
        value_t max_error = 0;
        for (int i = 0; i < a_columns; ++i)
        {
            for (int j = 0; j < b_rows; ++j)
            {
                value_t sum = 0;
                for (int k = 0; k < a_rows; ++k)
                    sum += A[k][i] * B[j][k];

                auto a = *(C.begin() + i * columns + j);

                max_error = std::fmax(max_error, std::fabs(sum - a));
            }
        }

#ifdef DEBUG_MATRIXMULTIPLICATION
        QUARISMA_LOGF(INFO, " =========== matrix multiplication =========");
        QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
        QUARISMA_LOGF(INFO, " ===========================================");
#else
        EXPECT_LE(max_error, tol);
#endif  // DEBUG_MATRIXMULTIPLICATION
    }
    else if (!transpose_a && transpose_b)
    {
        C                 = A * transpose(B);
        value_t max_error = 0;
        for (int i = 0; i < a_rows; ++i)
        {
            for (int j = 0; j < b_rows; ++j)
            {
                value_t sum = 0;
                for (int k = 0; k < a_columns; ++k)
                    sum += A[i][k] * B[j][k];

                auto a = *(C.begin() + i * columns + j);

                max_error = std::fmax(max_error, std::fabs(sum - a));
            }
        }
#ifdef DEBUG_MATRIXMULTIPLICATION
        QUARISMA_LOGF(INFO, " =========== matrix multiplication =========");
        QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
        QUARISMA_LOGF(INFO, " ===========================================");
#else
        EXPECT_LE(max_error, tol);
#endif  // DEBUG_MATRIXMULTIPLICATION
    }
    else if (transpose_a && !transpose_b)
    {
        C = transpose(A) * B;

        value_t max_error = 0;
        for (int i = 0; i < a_columns; ++i)
        {
            for (int j = 0; j < b_columns; ++j)
            {
                value_t sum = 0;
                for (int k = 0; k < a_rows; ++k)
                    sum += A[k][i] * B[k][j];

                auto a = *(C.begin() + i * columns + j);

                max_error = std::fmax(max_error, std::fabs(sum - a));
            }
        }
#ifdef DEBUG_MATRIXMULTIPLICATION
        QUARISMA_LOGF(INFO, " =========== matrix multiplication =========");
        QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
        QUARISMA_LOGF(INFO, " ===========================================");
#else
        EXPECT_LE(max_error, tol);
#endif  // DEBUG_MATRIXMULTIPLICATION
    }
    else
    {
        C = A * B;

        value_t max_error = 0;
        for (int i = 0; i < a_rows; ++i)
        {
            for (int j = 0; j < b_columns; ++j)
            {
                value_t sum = 0;
                for (int k = 0; k < a_columns; ++k)
                    sum += A[i][k] * B[k][j];

                auto a = *(C.begin() + i * columns + j);

                max_error = std::fmax(max_error, std::fabs(sum - a));
            }
        }
#ifdef DEBUG_MATRIXMULTIPLICATION
        QUARISMA_LOGF(INFO, " =========== matrix multiplication =========");
        QUARISMA_LOGF(INFO, " max error: %.2e ", max_error);
        QUARISMA_LOGF(INFO, " ===========================================");
#else
        EXPECT_LE(max_error, tol);
#endif  // DEBUG_MATRIXMULTIPLICATION
    }
}
}  // namespace

QUARISMATEST(Math, MatrixMultiplication)
{
    START_LOG_TO_FILE_NAME(MatrixMultiplication);

    const int rows    = 64 + 32 + 8 + 4 + 3;
    const int lda     = 64 + 3;
    const int columns = 64 + 32 + 8 + 4 + 1 + 9;

    matrix_vector_multiplication<double>(rows, columns);
    matrix_vector_multiplication<float>(rows, columns);

    vector_matrix_multiplication<float>(rows, columns);
    vector_matrix_multiplication<double>(rows, columns);

    matrix_multiplication<float>(rows, lda, columns, false, false);
    matrix_multiplication<float>(rows, lda, columns, true, false);
    matrix_multiplication<float>(rows, lda, columns, false, true);
    matrix_multiplication<float>(rows, lda, columns, true, true);

    matrix_multiplication<double>(rows, lda, columns, false, false);
    matrix_multiplication<double>(rows, lda, columns, true, false);
    matrix_multiplication<double>(rows, lda, columns, false, true);
    matrix_multiplication<double>(rows, lda, columns, true, true);

    END_LOG_TO_FILE_NAME(MatrixMultiplication);
    END_TEST();
}
