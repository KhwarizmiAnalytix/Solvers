#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "matrix_operation/matrix_inversion.h"
#include "terminals/matrix.h"
#include "quarismaTest.h"

namespace
{
template <typename>
struct Tolerance
{
};

template <>
struct Tolerance<float>
{
    static constexpr float value = 0.0002F;
};

template <>
struct Tolerance<double>
{
    static constexpr double value = 5.e-12;
};

template <typename matrix_type, typename value_t>
void test_inversion(size_t dim)
{
    matrix_type R(dim, dim);
    matrix_type Id(dim, dim);
    Id = 0.;

    for (size_t i = 0; i < dim; ++i)
    {
        Id[i][i] = 1.;
        R[i][i]  = 1.;
        for (size_t j = 0; j < i; ++j)
        {
            R[j][i] = 0;
            R[i][j] = static_cast<value_t>(0.4 * rand() / RAND_MAX);
        }
    }

    quarisma::vector<value_t> x_ref(dim);
    std::vector<quarisma_int> pivot(dim + 1);

    for (size_t i = 0; i < dim; i++)
        x_ref[i] = static_cast<value_t>(rand() / RAND_MAX);

    matrix_type A(dim, dim);
    {
        matrix_type t_R(dim, dim);
        t_R = transpose(R);
        A   = R * t_R;
    }
    matrix_type IA(dim, dim);
    IA.deepcopy(A);
    quarisma::matrix_invert(
        IA.data(), pivot.data(), dim, quarisma::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            IA[j][i] = IA[i][j];
        }
    }

    R                = IA * A;
    double max_error = hmax(fabs(R - Id));
    QUARISMA_LOGF(INFO, " CHOLESKY matrix_invert max error: %.2e ", max_error);
    EXPECT_LE(max_error, Tolerance<value_t>::value);

    IA.deepcopy(A);
    quarisma::matrix_invert(
        IA.data(), pivot.data(), dim, quarisma::linear_solver_type::LU_LINEAR_SOLVER);

    R         = IA * A;
    max_error = hmax(fabs(R - Id));
    QUARISMA_LOGF(INFO, " LU matrix_invert max error: %.2e ", max_error);
    EXPECT_LE(max_error, Tolerance<value_t>::value);

    quarisma::matrix_determinant(
        A.data(), pivot.data(), dim, quarisma::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    quarisma::matrix_determinant(
        A.data(), pivot.data(), dim, quarisma::linear_solver_type::LU_LINEAR_SOLVER);

    ASSERT_ANY_THROW(
        { quarisma::matrix_invert(IA.data(), pivot.data(), dim, (quarisma::linear_solver_type)5); };);

    ASSERT_ANY_THROW({
        quarisma::matrix_determinant(IA.data(), pivot.data(), dim, (quarisma::linear_solver_type)5);
    };);
}
}  // namespace

QUARISMATEST(Math, MatrixInversion)
{
    START_LOG_TO_FILE_NAME(MatrixInversion);

    const size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    test_inversion<quarisma::matrix<double>, double>(dim);
    test_inversion<quarisma::matrix<float>, float>(dim);

    test_inversion<quarisma::matrix<double>, double>(3);
    test_inversion<quarisma::matrix<float>, float>(3);

    END_LOG_TO_FILE_NAME(MatrixInversion);
    END_TEST();
}
