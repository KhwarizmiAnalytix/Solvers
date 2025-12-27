//#include <Eigen/Dense>
//#include<Eigen / LU>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "matrix_operation/linear_solver.h"
#include "matrix_operation/lu_decomposition.h"
#include "matrix_operation/matrix_inversion.h"
#include "terminals/matrix.h"
#include "quarismaTest.h"

namespace
{
template <typename matrix_type, typename value_t>
void test_lu(double tolerance)
{
    const size_t n = 11;

    std::vector<quarisma_int> pivot(n + 1);

    quarisma::vector<value_t> b(n);
    quarisma::vector<value_t> x(n);

    matrix_type A(n, n);

    matrix_type a1(n, n);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            A[i][j] = std::fabs(rand() / static_cast<value_t>(RAND_MAX));
        }
        A[i][i] = 1.;
        b[i]    = rand() / static_cast<value_t>(RAND_MAX);
    }

    a1.deepcopy(A);
    quarisma::lu_decomposition(a1.data(), n, pivot.data());

    //-----------------------------------------------------------------------------
    matrix_type L(n, n);
    matrix_type U(n, n);
    L = 0.;
    U = 0.;

    for (size_t i = 0; i < n; i++)
    {
        U[i][i] = a1[i][i];
        L[i][i] = 1.;
        for (size_t j = 0; j < i; j++)
        {
            L[i][j] = a1[i][j];
            U[j][i] = a1[j][i];
        }
    }
    matrix_type result(n, n);
    result = L * U;

#if defined(QUARISMA_LU_PIVOTING) || defined(QUARISMA_ENABLE_MKL)
    for (int i = n - 1; i >= 0; --i)
    {
        auto p = pivot[i] - 1;
        if (p != i)
        {
            for (int j = 0; j < n; ++j)
            {
                auto temp    = result[i][j];
                result[i][j] = result[p][j];
                result[p][j] = temp;
            }
        }
    }
#endif

    auto diff = hmax(fabs(result - A));
    QUARISMA_LOGF(INFO, "LU decompose max error %.1e", diff);
    EXPECT_LE(diff, tolerance);

    //-----------------------------------------------------------------------------
    matrix_type IA(n, n);
    IA.deepcopy(a1);

    quarisma::matrix_invert(
        IA.data(), pivot.data(), n, quarisma::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);
    quarisma::matrix_determinant(
        a1.data(), pivot.data(), n, quarisma::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);

    matrix_type Id(n, n);
    Id = 0.;
    for (size_t i = 0; i < n; i++)
        Id[i][i] = 1.;

    result = IA * A;
    result -= Id;

    diff = hmax(fabs(result));
    QUARISMA_LOGF(INFO, "LU decompose max error %.1e", diff);
    EXPECT_LE(diff, tolerance);

    //-----------------------------------------------------------------------------
    x.deepcopy(b);
    quarisma::linear_solver(
        a1.data(), pivot.data(), n, x.data(), quarisma::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);

    diff = quarisma::hmax(fabs(b - A * x));
    QUARISMA_LOGF(INFO, "LU linear max error %.1e", diff);
    EXPECT_LE(diff, tolerance);

    //-----------------------------------------------------------------------------
    //Eigen::Matrix2d Ae;
    //Ae << 2, 1, 2, 0.9999999999;
    //Eigen::FullPivLU<Eigen::Matrix2d> lu(Ae);
    //std::cout << "By default, the rank of A is found to be " << lu.rank() << std::endl;
    //lu.setThreshold(1e-5);
    //std::cout << "With threshold 1e-5, the rank of A is found to be " << lu.rank() << std::endl;
}
}  // namespace

QUARISMATEST(Math, LUDecomposition)
{
    START_LOG_TO_FILE_NAME(LUDecomposition);

    test_lu<quarisma::matrix<float>, float>(2.e-5);
    test_lu<quarisma::matrix<double>, double>(4.e-14);

    END_LOG_TO_FILE_NAME(LUDecomposition);
    END_TEST();
}
