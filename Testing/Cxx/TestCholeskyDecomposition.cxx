#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "AADTest.h"
#include "matrix_operation/cholesky_decomposition.h"
#include "matrix_operation/linear_solver.h"
#include "matrix_operation/matrix_inversion.h"
#include "terminals/matrix.h"
#include "quarismaTest.h"

#define TEST_AAD

namespace
{
template <typename value_t>
struct Tolerance
{
    static constexpr value_t value = (value_t)200 * std::numeric_limits<value_t>::epsilon();
};

namespace details
{
template <typename matrix_type, typename value_t>
matrix_type build_symetric_matrix(
    const std::vector<value_t>& values, size_t dim, quarisma::cholesky_decomposition_enum type)
{
    matrix_type M(dim, dim);
    size_t      offset = 0;
    switch (type)
    {
    case quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR:

        for (size_t i = 0; i < dim; ++i)
        {
            M[i][i] = values[offset++];
            for (size_t j = 0; j < i; ++j)
            {
                M[i][j] = values[offset++];
                M[j][i] = 0;
            }
        }
        break;

    default:

        for (size_t i = 0; i < dim; ++i)
        {
            M[i][i] = values[offset++];
            for (size_t j = 0; j < i; ++j)
            {
                M[j][i] = values[offset++];
                M[i][j] = 0;
            }
        }
        break;
    }

    return M;
};

template <typename matrix_type, typename value_t>
void build_cholesky_matrix(
    const std::vector<value_t>&         values,
    size_t                              dim,
    quarisma::cholesky_decomposition_enum type,
    bool                                test_aad)
{
    auto C = build_symetric_matrix<matrix_type, value_t>(values, dim, type);

    matrix_type A(dim, dim);
    matrix_type t_C(dim, dim);

    switch (type)
    {
    case quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR:
    {
        t_C = transpose(C);
        A   = C * t_C;
    }
    break;
    default:
    {
        t_C = transpose(C);
        A   = t_C * C;
    }
    break;
    }

    matrix_type R(dim, dim);

    R.deepcopy(A);
    quarisma::cholesky_decomposition(R.begin(), A.columns(), type);
    for (size_t i = 0; i < dim; ++i)
    {
        for (size_t j = i + 1; j < dim; ++j)
        {
            if (type == quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR)
            {
                R[i][j] = 0;
            }
            else
            {
                R[j][i] = 0;
            }
        }
    }
    auto diff = quarisma::hmax(fabs(R - C));

    QUARISMA_LOGF(INFO, "cholesky_decomposition max error %.1e", diff);
    EXPECT_LE(diff, Tolerance<value_t>::value);

    if (test_aad)
    {
        matrix_type R_aad(dim, dim);
        R_aad = 1.;

        matrix_type A_aad(dim, dim);
        A_aad = 0.;
        quarisma::cholesky_decomposition_aad(R_aad.begin(), R.begin(), dim, type, A_aad.begin());

        if constexpr (std::is_same_v<value_t, double>)
        {
            std::vector<value_t> state_parameters(dim * dim);
            state_parameters.assign(A.begin(), A.end());

            std::vector<value_t> state_parameters_aad(dim * dim);
            state_parameters_aad.assign(A_aad.begin(), A_aad.end());

            auto function = [dim, type](const std::vector<value_t>& x)
            {
                matrix_type A_tmp(x.data(), dim, dim);

                matrix_type R_tmp(dim, dim);
                R_tmp.deepcopy(A_tmp);
                quarisma::cholesky_decomposition(R_tmp.begin(), dim, type);
                for (size_t i = 0; i < dim; ++i)
                {
                    for (size_t j = i + 1; j < dim; ++j)
                    {
                        if (type == quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR)
                        {
                            R_tmp[i][j] = 0;
                        }
                        else
                        {
                            R_tmp[j][i] = 0;
                        }
                    }
                }

                return quarisma::accumulate(R_tmp);
            };

            EXPECT_TRUE(
                quarisma::aad_test::run_aad_test(
                    function, state_parameters, state_parameters_aad, false, true, false));
        }
    }

    switch (type)
    {
    case quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR:

        for (size_t i = 0; i < dim; ++i)
        {
            for (size_t j = 0; j < i; ++j)
            {
                R[j][i] = 0;
            }
        }
        break;
    case quarisma::cholesky_decomposition_enum::UPPER_TRIANGULAR:

        for (size_t i = 0; i < dim; ++i)
        {
            for (size_t j = 0; j < i; ++j)
            {
                R[i][j] = 0;
            }
        }
        break;
    }

    if (type == quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR)
    {
        std::vector<quarisma_int> pivot(dim);

        matrix_type IA(dim, dim);
        matrix_type RET(dim, dim);

        matrix_type Id(dim, dim);

        Id = 0.;
        for (size_t i = 0; i < dim; ++i)
        {
            Id[i][i] = 1.;
        }
        IA.deepcopy(R);
        quarisma::matrix_invert(
            IA.data(),
            pivot.data(),
            dim,
            quarisma::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

        for (size_t i = 0; i < dim; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                IA[j][i] = IA[i][j];
            }
        }

        RET = IA * A;

        diff = quarisma::hmax(fabs(Id - RET));
        QUARISMA_LOGF(INFO, "Cholesky decomposition matrix invert %.1e", diff);
        EXPECT_LE(diff, Tolerance<value_t>::value);

        IA.deepcopy(R);
        quarisma::matrix_determinant(
            IA.data(),
            pivot.data(),
            dim,
            quarisma::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

        quarisma::vector<value_t> x(dim);

        for (size_t i = 0; i < dim; i++)
        {
            x[i] = rand() / static_cast<value_t>(RAND_MAX);
        }
        quarisma::vector<value_t> b(dim);
        b.deepcopy(x);
        quarisma::linear_solver(
            R.begin(),
            pivot.data(),
            dim,
            x.data(),
            quarisma::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

        diff = quarisma::hmax(fabs(b - A * x));
        QUARISMA_LOGF(INFO, "Cholesky decomposition linear solver %.1e", diff);
        EXPECT_LE(diff, Tolerance<value_t>::value);
    }
};
}  // namespace details

template <typename matrix_type, typename value_t>
void test_cholseky(const size_t dim, bool test_aad)
{
    std::vector<value_t> values(dim * (dim + 1) / 2);

    size_t offset = 0;
    for (size_t i = 0; i < dim; ++i)
    {
        values[offset++] = static_cast<value_t>(5.);
        for (size_t j = 0; j < i; ++j)
        {
            values[offset++] = static_cast<value_t>(0.5 * rand() / RAND_MAX);
        }
    }

    QUARISMA_LOGF(INFO, "Cholesky decomposition LOWER_TRIANGULAR");
    details::build_cholesky_matrix<matrix_type, value_t>(
        values, dim, quarisma::cholesky_decomposition_enum::LOWER_TRIANGULAR, test_aad);

    QUARISMA_LOGF(INFO, "Cholesky decomposition UPPER_TRIANGULAR");
    details::build_cholesky_matrix<matrix_type, value_t>(
        values, dim, quarisma::cholesky_decomposition_enum::UPPER_TRIANGULAR, test_aad);

#ifndef QUARISMA_ENABLE_MKL
    ASSERT_ANY_THROW({
        quarisma::cholesky_decomposition(&values[0], 1, (quarisma::cholesky_decomposition_enum)'5');
    });

    ASSERT_ANY_THROW({
        quarisma::cholesky_decomposition_aad(
            &values[0], &values[0], 1, (quarisma::cholesky_decomposition_enum)'5', &values[0]);
    });
#endif
}
}  // namespace

QUARISMATEST(Math, CholeskyDecomposition)
{
    START_LOG_TO_FILE_NAME(CholeskyDecomposition);

    const size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    test_cholseky<quarisma::matrix<float>, float>(dim, false);
    test_cholseky<quarisma::matrix<double>, double>(dim, false);

    test_cholseky<quarisma::matrix<float>, float>(13, true);
    test_cholseky<quarisma::matrix<double>, double>(13, true);

    END_LOG_TO_FILE_NAME(CholeskyDecomposition);
    END_TEST();
}
