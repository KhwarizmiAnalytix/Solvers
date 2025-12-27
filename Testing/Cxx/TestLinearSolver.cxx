#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "matrix_operation/linear_solver.h"
#include "terminals/matrix.h"
#include "quarismaTest.h"

namespace
{
//template <typename matrix_type, typename value_t>
//void test_solve_tridiagonal()
//{
//    size_t  n    = 4;
//    value_t L[3] = {-1, -1, -1};    // NOLINT
//    value_t D[4] = {4, 4, 4, 4};    // NOLINT
//    value_t U[3] = {-1, -1, -1};    // NOLINT
//    value_t X[4] = {5, 5, 10, 23};  // NOLINT
//
//    value_t results[4] = {2, 3, 5, 7};  // NOLINT
//
//    quarisma::solve_tridiagonal(X, n, L, D, U);
//
//    value_t max_error = 0;
//    for (size_t i = 0; i < n; ++i)
//        max_error = std::fmax(std::fabs(X[i] - results[i]), max_error);
//
//    EXPECT_LE(max_error, std::numeric_limits<value_t>::epsilon());
//}

template <typename matrix_type, typename value_t>
void test_linear_solver(size_t dim, double tolerance)
{
    matrix_type R(dim, dim);

    for (size_t i = 0; i < dim; ++i)
    {
        R[i][i] = 1.;

        for (size_t j = 0; j < i; ++j)
        {
            R[i][j] = static_cast<value_t>(0.4 * rand() / RAND_MAX);
            R[j][i] = 0.;
        }
    }
    quarisma::vector<value_t> x_ref(dim);
    std::vector<quarisma_int> pivot(dim + 1);

    for (size_t i = 0; i < dim; i++)
        x_ref[i] = rand() / static_cast<value_t>(RAND_MAX);

    matrix_type A(dim, dim);
    {
        matrix_type t_R(dim, dim);
        t_R = transpose(R);
        A   = R * t_R;
    }

    quarisma::vector<value_t> b = A * x_ref;

    quarisma::vector<value_t> x(dim);
    x.deepcopy(b);
    quarisma::linear_solver(
        R.begin(),
        pivot.data(),
        dim,
        x.data(),
        quarisma::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);

    auto diff = hmax(fabs(x_ref - x));
    QUARISMA_LOGF(INFO, "Cholesky linear solver max error %.1e", diff);
    EXPECT_LE(diff, tolerance);

    x.deepcopy(b);
    R.deepcopy(A);
    quarisma::linear_solver(
        R.begin(), pivot.data(), dim, x.data(), quarisma::linear_solver_type::CHOLESKY_LINEAR_SOLVER);

    diff = hmax(fabs(x_ref - x));
    QUARISMA_LOGF(INFO, "Cholesky linear solver max error %.1e", diff);
    EXPECT_LE(diff, tolerance);

    x.deepcopy(b);
    R.deepcopy(A);
    quarisma::linear_solver(
        R.begin(), pivot.data(), dim, x.data(), quarisma::linear_solver_type::LU_LINEAR_SOLVER);

    diff = hmax(fabs(x_ref - x));
    QUARISMA_LOGF(INFO, "LU linear solver max error %.1e", diff);
    EXPECT_LE(diff, tolerance);
}
}  // namespace

QUARISMATEST(Math, LinearSolver)
{
    START_LOG_TO_FILE_NAME(LinearSolver);

    //test_solve_tridiagonal<quarisma::matrix<float>, float>();
    //test_solve_tridiagonal<quarisma::matrix<double>, double>();

    const size_t dim = 128 + 24 + 16 + 12 + 8 + 4 + 3 + 2 + 1;
    test_linear_solver<quarisma::matrix<float>, float>(dim, 1.e-2);
    test_linear_solver<quarisma::matrix<double>, double>(dim, 1.e-10);

    END_LOG_TO_FILE_NAME(LinearSolver);
    END_TEST();
}
