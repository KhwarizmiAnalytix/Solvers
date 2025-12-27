#include "quadrature/gaussian_quadrature.h"

#include <array>    // for array
#include <cmath>    // for fabs, sqrt, nan, pow
#include <cstdio>   // for fprintf, stderr
#include <cstdlib>  // for exit
#include <vector>   // for vector

#include "common/constants.h"
#include "terminals/vector.h"

namespace quarisma
{
namespace
{
//-----------------------------------------------------------------------------
bool gauss_hermite_precomputed(const size_t n, vector<double>& points, vector<double>& weights)
{
    switch (n)
    {
    case 2:
    {
        std::array<double, 2> tmp = {-0.70710678118654752440084436, 0.8862269254527580136490837};

        points[0]  = tmp[0];
        weights[0] = tmp[1];
    }
        return true;

    case 4:
    {
        std::array<double, 4> tmp = {
            -1.6506801238857845558833411,
            0.08131283544724517714303456,
            -0.52464762327529031788406025,
            0.8049140900055128365060492};
        points[0]  = tmp[0];
        points[1]  = tmp[2];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
    }
        return true;

    case 6:
    {
        std::array<double, 6> tmp = {
            -2.350604973674492222833922,
            0.004530009905508845640857473,
            -1.3358490740136969497148953,
            0.15706732032285664391631156,
            -0.43607741192761650867921595,
            0.7246295952243925240919147};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
    }
        return true;

    case 8:
    {
        std::array<double, 8> tmp = {
            -2.9306374202572440192235027,
            1.996040722113676192060905E-4,
            -1.9816567566958429258546306,
            0.017077983007413475456203056,
            -1.1571937124467801947207658,
            0.20780232581489187954325862,
            -0.38118699020732211685471889,
            0.66114701255824129103041597};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
    }
        return true;

    case 10:
    {
        std::array<double, 10> tmp = {
            -3.4361591188377376033267255,
            7.640432855232620629159368E-6,
            -2.5327316742327897964089608,
            0.0013436457467812326922015656,
            -1.7566836492998817734514012,
            0.0338743944554810631361647,
            -1.0366108297895136541774919,
            0.2401386110823146864165233,
            -0.34290132722370460878916503,
            0.610862633735325798783565};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        points[4]  = tmp[8];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
        weights[4] = tmp[9];
    }
        return true;

    case 12:
    {
        std::array<double, 12> tmp = {
            -3.8897248978697819192716427,
            2.658551684356301606023114E-7,
            -3.0206370251208897717106794,
            8.57368704358785865456906E-5,
            -2.2795070805010599001877286,
            0.003905390584629061859994384,
            -1.5976826351526047967096628,
            0.051607985615883929991873442,
            -0.94778839124016374370457813,
            0.26049231026416112923339614,
            -0.31424037625435911127661163,
            0.57013523626247957834711348};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        points[4]  = tmp[8];
        points[5]  = tmp[10];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
        weights[4] = tmp[9];
        weights[5] = tmp[11];
    }
        return true;

    case 14:
    {
        std::array<double, 14> tmp = {
            -4.3044485704736318126212981,
            8.62859116812515794532042E-9,
            -3.4626569336022705502089174,
            4.71648435501891674887689E-6,
            -2.7484707249854025686249985,
            3.550926135519236104836611E-4,
            -2.0951832585077168157349727,
            0.0078500547264579443104864433,
            -1.4766827311411408705835065,
            0.06850553422346520553871633,
            -0.87871378732939941611467931,
            0.27310560906424660335256919,
            -0.29174551067256207844611308,
            0.5364059097120901497949213};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        points[4]  = tmp[8];
        points[5]  = tmp[10];
        points[6]  = tmp[12];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
        weights[4] = tmp[9];
        weights[5] = tmp[11];
        weights[6] = tmp[13];
    }
        return true;

    case 16:
    {
        std::array<double, 16> tmp = {
            -4.6887389393058183646884986,
            2.654807474011182244709264E-10,
            -3.8694479048601226987194241,
            2.320980844865210653387494E-7,
            -3.1769991619799560268139946,
            2.711860092537881512018914E-5,
            -2.5462021578474813621593287,
            9.32284008624180529914277E-4,
            -1.9517879909162539774346554,
            0.0128803115355099736834643,
            -1.3802585391988807963720897,
            0.08381004139898582941542073,
            -0.8229514491446558925824545,
            0.28064745852853367536946334,
            -0.2734810461381524521582804,
            0.50792947901661374191351734};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        points[4]  = tmp[8];
        points[5]  = tmp[10];
        points[6]  = tmp[12];
        points[7]  = tmp[14];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
        weights[4] = tmp[9];
        weights[5] = tmp[11];
        weights[6] = tmp[13];
        weights[7] = tmp[15];
    }
        return true;

    case 18:
    {
        std::array<double, 18> tmp = {
            -5.0483640088744667683720376,
            7.828199772115891029251475E-12,
            -4.2481178735681264630234202,
            1.0467205795792082444355961E-8,
            -3.573769068486266079500676,
            1.810654481093430409597024E-6,
            -2.9613775055316068447786325,
            9.1811268679294035291467541E-5,
            -2.386299089166686000264593,
            0.001888522630268417894381753,
            -1.8355316042616288922538394,
            0.018640042387544651921931522,
            -1.3009208583896173656662656,
            0.09730174764131542933085372,
            -0.77668291926741166131665946,
            0.28480728566997957859560682,
            -0.2582677505190967592581161,
            0.48349569472545555287641052};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        points[4]  = tmp[8];
        points[5]  = tmp[10];
        points[6]  = tmp[12];
        points[7]  = tmp[14];
        points[8]  = tmp[16];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
        weights[4] = tmp[9];
        weights[5] = tmp[11];
        weights[6] = tmp[13];
        weights[7] = tmp[15];
        weights[8] = tmp[17];
    }
        return true;

    case 20:
    {
        std::array<double, 20> tmp = {-5.3874808900112328620169004,  2.229393645534151292522501E-13,
                                      -4.6036824495507442730776752,  4.399340992273180553628851E-10,
                                      -3.9447640401156252103756288,  1.086069370769281693999525E-7,
                                      -3.3478545673832163269149245,  7.802556478532063694145992E-6,
                                      -2.7888060584281304805250338,  2.28338636016353967257146E-4,
                                      -2.2549740020892755230823333,  0.003243773342237861832183247,
                                      -1.7385377121165862067808657,  0.024810520887463610882164953,
                                      -1.2340762153953230078858183,  0.10901720602002332001375503,
                                      -0.73747372854539435870560514, 0.28667550536283412971965971,
                                      -0.24534070830090124990383653, 0.46224366960061008965032864};

        points[0]  = tmp[0];
        points[1]  = tmp[2];
        points[2]  = tmp[4];
        points[3]  = tmp[6];
        points[4]  = tmp[8];
        points[5]  = tmp[10];
        points[6]  = tmp[12];
        points[7]  = tmp[14];
        points[8]  = tmp[16];
        points[9]  = tmp[18];
        weights[0] = tmp[1];
        weights[1] = tmp[3];
        weights[2] = tmp[5];
        weights[3] = tmp[7];
        weights[4] = tmp[9];
        weights[5] = tmp[11];
        weights[6] = tmp[13];
        weights[7] = tmp[15];
        weights[8] = tmp[17];
        weights[9] = tmp[19];
    }
        return true;

    default:
        return false;
    }
}
}  // namespace

//-----------------------------------------------------------------------------
void gaussian_quadrature::gauss_hermite_coefficients(
    const size_t    np,
    vector<double>& points,
    vector<double>& weights,
    bool            use_precomputed,
    double          tolerance,
    size_t          max_iter)
{
    if (use_precomputed)
    {
        use_precomputed = gauss_hermite_precomputed(np, points, weights);
    }

    const size_t m = (np + 1) / 2;
    if (!use_precomputed)
    {
        const auto n = static_cast<double>(np);

        const auto sqrt_2n = std::sqrt(2. * n);

        double z = 0.;
        for (size_t i = 0; i < m; i++)
        {
            switch (i)
            {
            case 0:
            {
                const auto tmp = std::sqrt(2. * n + 1.);
                z              = tmp - 1.85575 / std::cbrt(tmp);
            }
            break;
            case 1:
                z -= 1.14 * std::pow(n, 0.426) / z;
                break;
            case 2:
                z += 0.86 * (z + points[0]);
                break;
            case 3:
                z += .91 * (z + points[1]);
                break;
            default:
                z += z + points[i - 2];
            }

            double error = std::numeric_limits<double>::max();
            double p2    = 0.;
            for (size_t itr = 0; itr < max_iter; ++itr)
            {
                double p1 = constants::INVERSE_SQRT_SQRT_PI;
                for (size_t j = 0; j < np; j++)
                {
                    const auto tmp = static_cast<double>(j);
                    const auto p3  = p2;
                    p2             = p1;
                    p1 = (z * constants::SQRT_2 * p2 - std::sqrt(tmp) * p3) / std::sqrt(tmp + 1.);
                }

                p2 *= sqrt_2n;

                error = p1 / p2;
                z -= error;
                if (std::fabs(error) < tolerance)
                {
                    break;
                }
            }

            points[i]  = -z;
            weights[i] = 2. / (p2 * p2);
        }
    }

    for (size_t i = 0; i < m; i++)
    {
        points[np - 1 - i]  = -points[i];
        weights[np - 1 - i] = weights[i];
    }
}

//-----------------------------------------------------------------------------
void gaussian_quadrature::gauss_laguerre_coefficients(
    const double    alpha,
    const size_t    n,
    vector<double>& points,
    vector<double>& weights,
    double          tolerance,
    size_t          max_iter)
{
    QUARISMA_CHECK(
        alpha > -1., "Generalized Gauss Laguerre quadrature invalide for ", alpha, " < -1.");

    double     z   = 0.;
    const auto n_d = static_cast<double>(n);
    for (size_t i = 0; i < n; i++)
    {
        if (i == 0)
        {
            z = (1.0 + alpha) * (3.0 + 0.92 * alpha) /
                (1.0 + 2.4 * static_cast<double>(n) + 1.8 * alpha);
        }
        else if (i == 1)
        {
            z += (15.0 + 6.25 * alpha) / (1.0 + 0.9 * alpha + 2.5 * static_cast<double>(n));
        }
        else
        {
            const auto ai = static_cast<double>(i) - 1.;

            z += ((1.0 + 2.55 * ai) / (1.9 * ai) + 1.26 * ai * alpha / (1.0 + 3.5 * ai)) *
                 (z - points[i - 2]) / (1.0 + 0.3 * alpha);
        }

        double p2 = 0.;
        for (size_t itr = 0; itr < max_iter; ++itr)
        {
            double p1 = 1.0;
            p2        = 0.0;
            for (size_t j = 0; j < n; j++)
            {
                const auto j_d = static_cast<double>(j);

                const auto p = p2;
                p2           = p1;
                p1           = ((2. * j_d + 1. + alpha - z) * p2 - (j_d + alpha) * p) / (j_d + 1.);
            }

            auto error = (n_d * p1 - (n_d + alpha) * p2) / z;
            p2 *= error;
            error = p1 / error;

            z -= error;
            if (std::fabs(error) < tolerance)
            {
                break;
            }
        }
        points[i]  = z;
        weights[i] = -std::tgamma(alpha + n_d) / (std::tgamma(n_d) * n_d * p2);
    }
}

//-----------------------------------------------------------------------------
void gaussian_quadrature::gauss_legendre_coefficients(
    const size_t n, vector<double>& points, vector<double>& weights, double tolerance)
{
    double error;
    double p2;
    auto   n_d = static_cast<double>(n);

    const size_t m = (n + 1) / 2;
    for (size_t i = 0; i < m; i++)
    {
        double z = std::cos(
            constants::PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));

        do
        {
            auto p1 = 1.0;
            p2      = 0.0;
            for (size_t j = 0; j < n; j++)
            {
                auto       j_d = static_cast<double>(j);
                const auto p   = p2;
                p2             = p1;
                p1             = ((2.0 * j_d + 1.0) * z * p2 - j_d * p) / (j_d + 1);
            }

            p2 = n_d * (z * p1 - p2) / (z * z - 1.);

            error = p1 / p2;
            z -= error;
        } while (std::fabs(error) > tolerance);

        points[i]          = -z;
        points[n - 1 - i]  = z;
        weights[i]         = 2. / ((1. - z * z) * p2 * p2);
        weights[n - 1 - i] = weights[i];
    }
}

namespace
{
//-----------------------------------------------------------------------------
void kronrod_abscissa_and_weight(
    size_t        n,
    size_t        m,
    double        coef2,
    bool          even,
    const double* b,
    double&       x,
    double&       w,
    double        tolerance,
    size_t        max_iter)
{
    double fd = std::numeric_limits<double>::quiet_NaN();
    double d2 = std::numeric_limits<double>::quiet_NaN();

    bool success = (x == 0.0);
    /*
      Iterative process for the computation of a Kronrod abscissa.
    */
    for (size_t iter = 0; iter < max_iter; iter++)
    {
        double     b1 = 0.0;
        double     b2 = b[m];
        const auto yy = 4.0 * (x) * (x)-2.0;
        double     d1 = 0.0;

        double ai;
        double dif;
        if (even)
        {
            ai  = 2. * static_cast<double>(m) + 1.;
            d2  = ai * b[m];
            dif = 2.0;
        }
        else
        {
            ai  = static_cast<double>(m) + 1.;
            d2  = 0.0;
            dif = 1.0;
        }

        double b0 = b1;

        for (size_t k = 1; k <= m; k++)
        {
            ai            = ai - dif;
            size_t i      = m - k + 1;
            b0            = b1;
            b1            = b2;
            const auto d0 = d1;
            d1            = d2;
            b2            = yy * b1 - b0 + b[i - 1];
            if (!even)
            {
                ++i;
            }
            d2 = yy * d1 - d0 + ai * b[i - 1];
        }

        double f;
        if (even)
        {
            f  = (x) * (b2 - b1);
            fd = d2 + d1;
        }
        else
        {
            f  = 0.5 * (b2 - b0);
            fd = 4.0 * (x)*d2;
        }
        /*
          Newton correction.
        */
        const auto delta = f / fd;
        x                = x - delta;

        if (success)
        {
            break;
        }

        if (std::fabs(delta) <= tolerance)
        {
            success = true;
        }
    }

    QUARISMA_CHECK(success, "ABWE2 - Fatal error! Iteration limit reached");

    /*
      Computation of the weight.
    */
    double d0 = 1.0;
    double d1 = x;
    double ai = 0.0;
    for (size_t k = 2; k <= n; k++)
    {
        ai = ai + 1.0;
        d2 = ((ai + ai + 1.0) * (x)*d1 - ai * d0) / (ai + 1.0);
        d0 = d1;
        d1 = d2;
    }

    w = coef2 / (fd * d2);
}

//-----------------------------------------------------------------------------
void gaussian_abscissa_and_two_weights(
    size_t        n,
    size_t        m,
    double        coef2,
    bool          even,
    const double* b,
    double&       x,
    double&       w1,
    double&       w2,
    double        tolerance,
    size_t        max_iter)
{
    bool success = (x == 0.0);
    /*
      Iterative process for the computation of a Gaussian abscissa.
    */

    double pd2 = std::numeric_limits<double>::quiet_NaN();
    double p0  = std::numeric_limits<double>::quiet_NaN();

    for (size_t iter = 0; iter < max_iter; iter++)
    {
        p0         = 1.0;
        double p1  = x;
        double pd0 = 0.0;
        double pd1 = 1.0;
        /*
          When N is 1, we need to initialize P2 and PD2 to avoid problems with FIXED_STRIKE.
        */
        double p2 = std::numeric_limits<double>::quiet_NaN();
        if (n <= 1)
        {
            /*if (!quarisma::is_almost_zero(x))
            {
                p2  = (3.0 * x * x - 1.0) / 2.0;
                pd2 = 3.0 * x;
            }
            else
            {*/
            p2  = 3.0 * x;
            pd2 = 3.0;
            //}
        }

        double ai = 0.0;
        for (size_t k = 2; k <= n; k++)
        {
            ai  = ai + 1.0;
            p2  = ((ai + ai + 1.0) * (x)*p1 - ai * p0) / (ai + 1.0);
            pd2 = ((ai + ai + 1.0) * (p1 + (x)*pd1) - ai * pd0) / (ai + 1.0);
            p0  = p1;
            p1  = p2;
            pd0 = pd1;
            pd1 = pd2;
        }
        /*
          Newton correction.
        */
        const auto delta = p2 / pd2;

        x -= delta;

        if (success)
        {
            break;
        }

        if (std::fabs(delta) <= tolerance)
        {
            success = true;
        }
    }

    QUARISMA_CHECK(success, "ABWE2 - Fatal error! Iteration limit reached");

    /*
      Computation of the weight.
    */
    const auto an = n;

    w2 = 2.0 / (static_cast<double>(an) * pd2 * p0);

    double     p1 = 0.0;
    double     p2 = b[m];
    const auto yy = 4.0 * (x) * (x)-2.0;
    for (size_t k = 1; k <= m; k++)
    {
        const auto i = m - k;
        p0           = p1;
        p1           = p2;
        p2           = yy * p1 - p0 + b[i];
    }

    w1 = even ? w2 + coef2 / (pd2 * (x) * (p2 - p1)) : w2 + 2.0 * coef2 / (pd2 * (p2 - p0));
}
}  // namespace

//-----------------------------------------------------------------------------
void gaussian_quadrature::gauss_kronrod(
    size_t          n,
    vector<double>& x,
    vector<double>& w1,
    vector<double>& w2,
    double          tolerance,
    size_t          max_iter)
{
    std::vector<double> b(((n + 1) / 2) + 1);
    std::vector<double> tau((n + 1) / 2);

    const auto m = (n + 1) / 2;

    const auto even = (2 * m == n);

    double d  = 2.0;
    double an = 0.0;
    for (size_t k = 0; k < n; k++)
    {
        ++an;
        d = d * an / (an + 0.5);
    }
    /*
      Calculation of the Chebyshev coefficients of the orthogonal polynomial.
    */
    tau[0]    = (an + 2.0) / (an + an + 3.0);
    b[m - 1]  = tau[0] - 1.0;
    double ak = an;

    for (size_t l = 1; l < m; l++)
    {
        ak     = ak + 2.0;
        tau[l] = ((ak - 1.0) * ak - an * (an + 1.0)) * (ak + 2.0) * tau[l - 1] /
                 (ak * ((ak + 3.0) * (ak + 2.0) - an * (an + 1.0)));
        b[m - l - 1] = tau[l];

        for (size_t ll = 1; ll <= l; ll++)
        {
            b[m - l - 1] = b[m - l - 1] + tau[ll - 1] * b[m - l + ll - 1];
        }
    }

    b[m] = 1.0;
    /*
      Calculation of approximate values for the abscissas.
    */
    double     bb   = sin(1.570796 / (an + an + 1.0));
    double     x1   = sqrt(1.0 - bb * bb);
    const auto s    = 2.0 * bb * x1;
    const auto c    = sqrt(1.0 - s * s);
    const auto coef = 1.0 - (1.0 - 1.0 / an) / (8.0 * an * an);
    double     xx   = coef * x1;
    /*
      Coefficient needed for weights.

      COEF2 = 2^(2*n+1) * n! * n! / (2n+1)!
    */
    double coef2 = 2.0 / (double)(2 * n + 1);
    for (size_t i = 1; i <= n; i++)
    {
        coef2 = coef2 * 4.0 * (double)(i) / (double)(n + i);
    }
    /*
      Calculation of the K-th abscissa (a Kronrod abscissa) and the
      corresponding weight.
    */
    for (size_t k = 0; k < n; k += 2)
    {
        kronrod_abscissa_and_weight(n, m, coef2, even, b.data(), xx, w1[k], tolerance, max_iter);
        w2[k] = 0.0;

        x[k]     = xx;
        double y = x1;
        x1       = y * c - bb * s;
        bb       = y * s + bb * c;

        if (k == n - 1)
        {
            xx = 0.0;
        }
        else
        {
            xx = coef * x1;
        }
        /*
          Calculation of the K+1 abscissa (a Gaussian abscissa) and the
          corresponding weights.
        */
        gaussian_abscissa_and_two_weights(
            n, m, coef2, even, b.data(), xx, w1[k + 1], w2[k + 1], tolerance, max_iter);

        x[k + 1] = xx;
        y        = x1;
        x1       = y * c - bb * s;
        bb       = y * s + bb * c;
        xx       = coef * x1;
    }
    /*
      If N is even, we have one more Kronrod abscissa to compute,
      namely the origin.
    */
    if (even)
    {
        xx = 0.0;
        kronrod_abscissa_and_weight(n, m, coef2, even, b.data(), xx, w1[n], tolerance, max_iter);
        w2[n] = 0.0;
        x[n]  = xx;
    }
}
}  // namespace quarisma
