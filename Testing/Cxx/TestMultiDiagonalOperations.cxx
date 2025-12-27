#include <cmath>
#include <cstddef>

#include "AADTest.h"
#include "common/pentadiagonal_operations.h"
#include "common/tridiagonal_operations.h"
#include "randomgen/mersenne_twister.h"
#include "quarismaTest.h"

using namespace quarisma;

namespace
{
void test_multidiagonal_solver(size_t stencil, size_t nx, size_t ny, size_t nz)
{
    const size_t n = nx * ny * nz;

    mersenne_twister m(1, 0);

    matrix<double> mat(stencil, n);
    m.gaussians(mat.data(), mat.size(), 0);

    matrix<double> decomposed(stencil, n);
    decomposed.deepcopy(mat);
    matrix<double> decomposed_parllel(stencil, n);
    decomposed_parllel.deepcopy(mat);

    const double time_multiplier = 1.0;

    if (stencil == 5)
    {
        pentadiagonal_operations::decomposition(decomposed, nx, ny, nz, false);
        pentadiagonal_operations::decomposition(decomposed_parllel, nx, ny, nz, true);
    }
    else
    {
        tridiagonal_operations::decomposition(decomposed, nx, ny, nz, false);
        tridiagonal_operations::decomposition(decomposed_parllel, nx, ny, nz, true);
    }
    QUARISMA_LOGF(INFO, "decomposition error:    %f  ", hmax(fabs(decomposed_parllel - decomposed)));

    vector<double> result(n);
    m.gaussians(result.data(), n, 0);

    vector<double> tmp(n);
    vector<double> tmp2(n);
    vector<double> tmp_parllel(n);
    vector<double> tmp2_parllel(n);
    tmp2         = 0.;
    tmp2_parllel = 0.;

    if (stencil == 5)
    {
        tmp.deepcopy(result);
        pentadiagonal_operations::solve_decomposed(tmp, decomposed, nx, ny, nz, false);
        pentadiagonal_operations::multiply(tmp2, tmp, mat, nx, ny, nz, time_multiplier, false);

        tmp_parllel.deepcopy(result);
        pentadiagonal_operations::solve_decomposed(
            tmp_parllel, decomposed_parllel, nx, ny, nz, true);
        pentadiagonal_operations::multiply(
            tmp2_parllel, tmp_parllel, mat, nx, ny, nz, time_multiplier, true);
    }
    else
    {
        tmp.deepcopy(result);
        tridiagonal_operations::solve_decomposed(tmp, decomposed, nx, ny, nz, false);
        tridiagonal_operations::multiply(tmp2, tmp, mat, nx, ny, nz, time_multiplier, false);

        tmp_parllel.deepcopy(result);
        tridiagonal_operations::solve_decomposed(tmp_parllel, decomposed_parllel, nx, ny, nz, true);
        tridiagonal_operations::multiply(
            tmp2_parllel, tmp_parllel, mat, nx, ny, nz, time_multiplier, true);
    }
    QUARISMA_LOG_INFO("solve_decomposed error: " << hmax(fabs(tmp_parllel - tmp)));
    QUARISMA_LOG_INFO("multiply error:         " << hmax(fabs(tmp2_parllel - tmp2)));
    QUARISMA_LOG_INFO("single error:           " << hmax(fabs(tmp2 - result)));
    QUARISMA_LOG_INFO("parallel error:         " << hmax(fabs(tmp2_parllel - result)));

    auto diff = hmax(fabs(tmp2 - result));
    EXPECT_LE(diff, 5.E-11);

    if (stencil == 3)
    {
        {
            decomposed.deepcopy(mat);
            vector<double> x(n);
            x.deepcopy(result);
            //Ax=b
            tridiagonal_operations::decomposition(decomposed, nx, ny, nz);
            tridiagonal_operations::solve_decomposed(x, decomposed, nx, ny, nz);

            vector<double> x_aad(n);
            x_aad = 1.;
            matrix<double> decomposed_aad(stencil, n);
            decomposed_aad = 0.;

            //x=A*b
            tridiagonal_operations::solve_decomposed_aad(
                decomposed_aad, x_aad, x, decomposed, nx, ny, nz);
            tridiagonal_operations::decomposition_aad(decomposed_aad, decomposed, nx, ny, nz);

            //AAD testing:
            std::vector<double> state_parameters;
            state_parameters.reserve(n + stencil * n);
            state_parameters.assign(result.begin(), result.end());
            state_parameters.insert(state_parameters.end(), mat.begin(), mat.end());

            std::vector<double> state_parameters_aad;
            state_parameters_aad.reserve(n + stencil * n);
            state_parameters_aad.assign(x_aad.begin(), x_aad.end());
            state_parameters_aad.insert(
                state_parameters_aad.end(), decomposed_aad.begin(), decomposed_aad.end());

            auto function =
                [&decomposed, &tmp, nx, ny, nz, stencil, n](const std::vector<double>& x)
            {
                vector<double> X(x.data(), n);
                tmp.deepcopy(X);

                matrix<double> m(x.data() + n, stencil, n);
                decomposed.deepcopy(m);

                tridiagonal_operations::decomposition(decomposed, nx, ny, nz);
                tridiagonal_operations::solve_decomposed(tmp, decomposed, nx, ny, nz);

                return quarisma::accumulate(tmp);
            };

            EXPECT_TRUE(
                quarisma::aad_test::run_aad_test(
                    function, state_parameters, state_parameters_aad, false, false, false));
        }
        {
            tmp2 = 0.;
            tmp.deepcopy(result);
            tridiagonal_operations::multiply(tmp2, tmp, mat, nx, ny, nz, time_multiplier, false);

            matrix<double> mat_aad(stencil, n);
            vector<double> tmp_aad(n);
            vector<double> tmp2_aad(n);

            mat_aad  = 0.;
            tmp_aad  = 0.;
            tmp2_aad = 1.;
            tridiagonal_operations::multiply_aad(
                mat_aad,
                tmp_aad,
                tmp2_aad,
                tmp,
                mat,
                nx,
                ny,
                nz,
                time_multiplier,
                false);  //AAD testing:

            std::vector<double> state_parameters;
            state_parameters.reserve(n + stencil * n);
            state_parameters.assign(result.begin(), result.end());
            state_parameters.insert(state_parameters.end(), mat.begin(), mat.end());

            std::vector<double> state_parameters_aad;
            state_parameters_aad.reserve(n + stencil * n);
            state_parameters_aad.assign(tmp_aad.begin(), tmp_aad.end());
            state_parameters_aad.insert(state_parameters_aad.end(), mat_aad.begin(), mat_aad.end());

            auto function =
                [&tmp2, nx, ny, nz, stencil, n, time_multiplier](const std::vector<double>& x)
            {
                tmp2 = 0.;
                vector<double> X(x.data(), n);
                matrix<double> m(x.data() + n, stencil, n);

                tridiagonal_operations::multiply(tmp2, X, m, nx, ny, nz, time_multiplier, false);

                return quarisma::accumulate(tmp2);
            };

            EXPECT_TRUE(
                quarisma::aad_test::run_aad_test(
                    function, state_parameters, state_parameters_aad, false, false, false));
        }
    }
}
}  // namespace

QUARISMATEST(Math, MultiDiagonalOperations)
{
    START_LOG_TO_FILE_NAME(MultiDiagonalOperations);

    size_t nx = 8;
    size_t ny = 4;
    size_t nz = 5;

    test_multidiagonal_solver(3, 1, nx * nz * ny, 1);
    test_multidiagonal_solver(3, nx, ny, nz);
    test_multidiagonal_solver(3, nx * nz, ny, 1);
    test_multidiagonal_solver(3, 1, ny, nx * nz);

    test_multidiagonal_solver(5, nx, ny, nz);
    test_multidiagonal_solver(5, nx * nz, ny, 1);
    test_multidiagonal_solver(5, 1, ny, nx * nz);

    END_LOG_TO_FILE_NAME(MultiDiagonalOperations);
    END_TEST();
}
