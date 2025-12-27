#include "common/finite_difference.h"

#include "smp/tools.h"

namespace quarisma::finite_difference
{
//-----------------------------------------------------------------------------
void nonuniform_five_points_derivative_0(
    const double* x, size_t i, double*& convection, double*& diffusion)
{
    bool is_zero = (i == 0);
    int  shift_1 = is_zero ? 1 : -1;
    int  shift_2 = is_zero ? 2 : -2;

    const auto x_0 = x[i];
    const auto x_1 = x[i + shift_1];
    const auto x_2 = x[i + shift_2];
    const auto m_1 = 1. / ((x_1 - x_0) * (x_1 - x_2));
    const auto m_2 = 1. / ((x_2 - x_0) * (x_2 - x_1));

    size_t l0 = is_zero ? 0 : 4;
    size_t l1 = is_zero ? 1 : 3;
    size_t l3 = is_zero ? 3 : 1;
    size_t l4 = is_zero ? 4 : 0;
    if (convection != nullptr)
    {
        convection[l0] = 0.;
        convection[l1] = 0.;
        convection[2]  = 1. / (x_0 - x_1) + 1. / (x_0 - x_2);
        convection[l3] = (x_0 - x_2) * m_1;
        convection[l4] = (x_0 - x_1) * m_2;

        convection += 5;
    }

    if (diffusion != nullptr)
    {
        diffusion[l0] = 0.;
        diffusion[l1] = 0.;
        diffusion[2]  = 2. / ((x_0 - x_1) * (x_0 - x_2));
        diffusion[l3] = 2. * m_1;
        diffusion[l4] = 2. * m_2;

        diffusion += 5;
    }
};

//-----------------------------------------------------------------------------
void nonuniform_five_points_derivative_1(
    const double* x, size_t i, double*& convection, double*& diffusion)
{
    bool is_one  = (i == 1);
    int  shift_1 = is_one ? 1 : -1;
    int  shift_2 = is_one ? 2 : -2;

    const auto x_m_1 = x[i - shift_1];
    const auto x_0   = x[i];
    const auto x_p_1 = x[i + shift_1];
    const auto x_p_2 = x[i + shift_2];

    const auto m_1 = 1. / ((x_m_1 - x_0) * (x_m_1 - x_p_1) * (x_m_1 - x_p_2));
    const auto m_2 = 1. / ((x_p_1 - x_m_1) * (x_p_1 - x_0) * (x_p_1 - x_p_2));
    const auto m_3 = 1. / ((x_p_2 - x_m_1) * (x_p_2 - x_0) * (x_p_2 - x_p_1));

    const size_t l0 = is_one ? 0 : 4;
    const size_t l1 = is_one ? 1 : 3;
    const size_t l3 = is_one ? 3 : 1;
    const size_t l4 = is_one ? 4 : 0;

    if (convection != nullptr)
    {
        convection[l0] = 0.;
        convection[l1] = (x_0 - x_p_1) * (x_0 - x_p_2) * m_1;
        convection[2]  = 1. / (x_0 - x_m_1) + 1. / (x_0 - x_p_1) + 1. / (x_0 - x_p_2);
        convection[l3] = (x_0 - x_m_1) * (x_0 - x_p_2) * m_2;
        convection[l4] = (x_0 - x_m_1) * (x_0 - x_p_1) * m_3;

        convection += 5;
    }
    if (diffusion != nullptr)
    {
        diffusion[l0] = 0.;
        diffusion[l1] = 2. * ((x_0 - x_p_1) + (x_0 - x_p_2)) * m_1;
        diffusion[2] =
            2. * (1. / ((x_0 - x_m_1) * (x_0 - x_p_1)) + 1. / ((x_0 - x_m_1) * (x_0 - x_p_2)) +
                  1. / ((x_0 - x_p_1) * (x_0 - x_p_2)));
        diffusion[l3] = 2. * ((x_0 - x_m_1) + (x_0 - x_p_2)) * m_2;
        diffusion[l4] = 2. * ((x_0 - x_m_1) + (x_0 - x_p_1)) * m_3;

        diffusion += 5;
    }
};

//-----------------------------------------------------------------------------
void nonuniform_five_points_derivative_mid(
    const double* x, size_t i, double*& convection, double*& diffusion, bool upwind)
{
    const auto x_i = x[i];

    const auto h_up_2   = x_i - x[i + 2];
    const auto h_up_1   = x_i - x[i + 1];
    const auto h_down_1 = x_i - x[i - 1];
    const auto h_down_2 = x_i - x[i - 2];

    if (convection != nullptr)
    {
        if (upwind)
        {
            *convection++ = 0.;
            *convection++ =
                -h_up_1 * h_up_2 / (h_down_1 * (h_down_1 - h_up_1) * (h_down_1 - h_up_2));
            *convection++ = 1. / h_down_1 + 1. / h_up_1 + 1. / h_up_2;
            *convection++ = -h_down_1 * h_up_2 / ((h_up_1 - h_down_1) * h_up_1 * (h_up_1 - h_up_2));
            *convection++ = -h_down_1 * h_up_1 / ((h_up_2 - h_down_1) * h_up_2 * (h_up_2 - h_up_1));
        }
        else
        {
            const auto a = h_up_2 * h_up_1 / (h_down_2 - h_down_1);
            const auto b = h_down_1 * h_down_2 / (h_up_1 - h_up_2);

            *convection++ = a * h_down_1 / (h_down_2 * (h_down_2 - h_up_1) * (h_down_2 - h_up_2));
            *convection++ = -a * h_down_2 / (h_down_1 * (h_down_1 - h_up_1) * (h_down_1 - h_up_2));
            *convection++ = 1. / h_down_2 + 1. / h_down_1 + 1. / h_up_1 + 1. / h_up_2;
            *convection++ = b * h_up_2 / (h_up_1 * (h_up_1 - h_down_1) * (h_up_1 - h_down_2));
            *convection++ = -b * h_up_1 / (h_up_2 * (h_up_2 - h_down_1) * (h_up_2 - h_down_2));
        }
    }

    if (diffusion != nullptr)
    {
        auto a = (h_up_2 + h_up_1);
        auto b = h_up_2 * h_up_1;

        *diffusion++ =
            2. * (h_down_1 * a + b) /
            (h_down_2 * (h_down_2 - h_down_1) * (h_down_2 - h_up_1) * (h_down_2 - h_up_2));

        *diffusion++ =
            2. * (h_down_2 * a + b) /
            ((h_down_1 - h_down_2) * h_down_1 * (h_down_1 - h_up_1) * (h_down_1 - h_up_2));

        *diffusion++ = 2. / (h_down_2 * h_down_1) + 2. / (h_down_2 * h_up_1) +
                       2. / (h_down_2 * h_up_2) + 2. / (h_down_1 * h_up_1) +
                       2. / (h_down_1 * h_up_2) + 2. / (h_up_1 * h_up_2);

        a = (h_down_2 + h_down_1);
        b = h_down_2 * h_down_1;

        *diffusion++ = 2. * (h_up_2 * a + b) /
                       ((h_up_1 - h_down_2) * (h_up_1 - h_down_1) * h_up_1 * (h_up_1 - h_up_2));

        *diffusion++ = 2. * (h_up_1 * a + b) /
                       ((h_up_2 - h_down_2) * (h_up_2 - h_down_1) * (h_up_2 - h_up_1) * h_up_2);
    }
};

//-----------------------------------------------------------------------------
void nonuniform_three_points_derivative_mid(
    const double* x, size_t i, double*& convection, double*& diffusion, bool upwind)
{
    double x_i = x[i];

    double h_up   = x[i + 1] - x_i;
    double h_down = x_i - x[i - 1];

    const auto h_up_2   = h_up * h_up;
    const auto h_down_2 = h_down * h_down;

    auto l = 1. / (h_down_2 * h_up + h_up_2 * h_down);

    if (convection != nullptr)
    {
        if (upwind)
        {
            *convection++ = 0.;
            *convection++ = -1. / h_up;
            *convection++ = 1. / h_up;
        }
        else
        {
            *convection++ = -h_up_2 * l;
            *convection++ = -(h_down_2 - h_up_2) * l;
            *convection++ = h_down_2 * l;
        }
    }
    if (diffusion != nullptr)
    {
        l *= 2.;
        *diffusion++ = h_up * l;
        *diffusion++ = -(h_down + h_up) * l;
        *diffusion++ = h_down * l;
    }
}

//-----------------------------------------------------------------------------
void nonuniform_derivation(
    const double* x,
    size_t        nx,
    double*&      convection_ptr,
    double*&      diffusion_ptr,
    bool          upwind,
    size_t        stencil)
{
    if (stencil == 3)
    {
        double h = 1. / (x[1] - x[0]);

        *convection_ptr++ = 0.;
        *convection_ptr++ = -h;
        *convection_ptr++ = h;

        *diffusion_ptr++ = 0.;
        *diffusion_ptr++ = 0.;
        *diffusion_ptr++ = 0.;

        for (size_t i = 1; i < nx - 1; ++i)
        {
            finite_difference::nonuniform_three_points_derivative_mid(
                x, i, convection_ptr, diffusion_ptr, upwind);
        }

        h = 1. / (x[nx - 1] - x[nx - 2]);

        *convection_ptr++ = -h;
        *convection_ptr++ = h;
        *convection_ptr++ = 0.;

        *diffusion_ptr++ = 0.;
        *diffusion_ptr++ = 0.;
        *diffusion_ptr++ = 0.;
    }
    else
    {
        finite_difference::nonuniform_five_points_derivative_0(x, 0, convection_ptr, diffusion_ptr);

        finite_difference::nonuniform_five_points_derivative_1(x, 1, convection_ptr, diffusion_ptr);

        for (size_t i = 2; i < nx - 2; ++i)
        {
            finite_difference::nonuniform_five_points_derivative_mid(
                x, i, convection_ptr, diffusion_ptr, upwind);
        }

        finite_difference::nonuniform_five_points_derivative_1(
            x, nx - 2, convection_ptr, diffusion_ptr);

        finite_difference::nonuniform_five_points_derivative_0(
            x, nx - 1, convection_ptr, diffusion_ptr);
    }
}
}  // namespace quarisma::finite_difference
