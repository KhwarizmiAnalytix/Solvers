#if defined(_MSC_VER) && !defined(QUARISMA_DISPLAY_WIN32_WARNINGS)
#pragma warning(push)
#pragma warning(disable : 4305)
#endif  // _MSC_VER

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "matrix_operation/svd_decomposition.h"
#include "terminals/matrix.h"
#include "util/logger.h"
#include "quarismaTest.h"

namespace
{
template <typename>
struct SVDTolerance
{
};

template <>
struct SVDTolerance<float>
{
    static constexpr double tolerance = 2.e-4;
};

template <>
struct SVDTolerance<double>
{
    static constexpr double tolerance = 5.e-13;
};

template <typename value_t>
int test_svd(const size_t rows, const size_t columns)
{
    quarisma::matrix<value_t> M(rows, columns);

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < columns; j++)
            M[i][j] = static_cast<value_t>(20. * rand() / RAND_MAX);

    size_t ldu = (columns < rows ? columns : rows);

    std::vector<value_t>    D(ldu);
    quarisma::matrix<value_t> U(rows, ldu);
    quarisma::matrix<value_t> tV(ldu, columns);

    const auto ldm = columns;
    const auto ldv = columns;
    quarisma::svd_decomposition(
        rows, columns, M.begin(), ldm, D.data(), U.begin(), ldu, tV.begin(), ldv);

    {
        value_t max_error = 0;

        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < columns; j++)
            {
                value_t sum = M[i][j];
                for (size_t k = 0; k < ldu; k++)
                    sum -= U[i][k] * D[k] * tV[k][j];

                max_error = std::fmax(std::fabs(sum), max_error);
            }

        QUARISMA_LOG_INFO("SVD max error " << max_error);
        // QUARISMA_LOG_INFO( "M matrix  \n" << M);
        // QUARISMA_LOG_INFO( "U matrix  \n" << U);
        // QUARISMA_LOG_INFO( "tV matrix \n" << tV);

        EXPECT_LT(max_error, SVDTolerance<value_t>::tolerance);
    }

    return quarisma::quarismaTesting::PASSED;
}
}  // namespace

QUARISMATEST(Math, SVDDecomposition)
{
    START_LOG_TO_FILE_NAME(SVDDecomposition);

    size_t rows    = 6;
    size_t columns = 5;
    test_svd<float>(rows, columns);
    test_svd<double>(rows, columns);

    rows    = 5;
    columns = 6;
    test_svd<float>(rows, columns);
    test_svd<double>(rows, columns);

    rows    = 25;
    columns = 11;
    test_svd<float>(rows, columns);
    test_svd<double>(rows, columns);

    END_LOG_TO_FILE_NAME(SVDDecomposition);
    END_TEST();
}
#if defined(_MSC_VER) && !defined(QUARISMA_DISPLAY_WIN32_WARNINGS)
#pragma warning(pop)
#endif
