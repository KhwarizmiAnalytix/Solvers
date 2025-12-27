#include "common/runge_kutta_solver.h"

#include <cmath>    // for fabs, pow, isnan
#include <cstddef>  // for size_t

#include "util/exception.h"  // for QUARISMA_CHECK, QUARISMA_CHECK_DEBUG

namespace quarisma
{
//-----------------------------------------------------------------------------
inline void rkck(
    double&                                 yout,
    double&                                 yerr,
    const double                            y,
    const double                            dydx,
    const double                            tmp,
    const double                            h,
    const runge_kutta_solver::ODE_function& derivs)
{
    // first step
    yout = y + 0.2 * h * dydx;

    // second step
    const auto ak2 = derivs(tmp + 0.2 * h, yout);
    yout           = y + h * 0.075 * (dydx + 3. * ak2);

    // third step
    const auto ak3 = derivs(tmp + 0.3 * h, yout);
    yout           = y + h * 0.3 * (dydx - 0.3 * ak2 + 4. * ak3);

    // fourth step
    const auto ak4 = derivs(tmp + 0.6 * h, yout);
    yout           = y + h * (2.5 * ak2 + (35.0 * ak4 - 70.0 * ak3 - 5.5 * dydx) / 27.);

    // fifth step
    const auto ak5 = derivs(tmp + h, yout);
    yout = y + h * (0.0294958043981481 * dydx + 0.341796875 * ak2 + 0.0415943287037037 * ak3 +
                    0.4003454137731480 * ak4 + 0.061767578125 * ak5);

    // sixth step
    const auto ak6 = derivs(tmp + 0.875 * h, yout);

    yout = y + h * (0.0978835978835979 * dydx + 0.402576489533011 * ak3 + 0.21043771043771 * ak4 +
                    0.28910220214568 * ak6);

    yerr = h * (0.0186685860938579 * ak3 - 0.004293774801587310 * dydx - 0.0341550268308081 * ak4 -
                0.0193219866071429 * ak5 + 0.0391022021456804 * ak6);
}

//-----------------------------------------------------------------------------
void rkqs(
    double&                                 y,
    double&                                 x,
    double&                                 hdid,
    double&                                 hnext,
    const double                            dydx,
    const double                            htry,
    const double                            tolerance,
    const double                            yScale,
    const runge_kutta_solver::ODE_function& derivs,
    const double                            adaptiverk_safety,
    const double                            adaptiverk_pgrow,
    const double                            adaptiverk_pshrink,
    const double                            adaptiverk_errcon)
{
    double xnew;
    double yerr;
    double ytemp;

    double h = htry;

    for (;;)
    {
        rkck(ytemp, yerr, y, dydx, x, h, derivs);

        const auto errmax = std::fabs(yerr / (tolerance * yScale));

        if (errmax > 1.)
        {
            const auto htemp1 = adaptiverk_safety * h * std::pow(errmax, adaptiverk_pshrink);
            const auto htemp2 = h / 10;

            h =
                ((h >= 0.0)        ? htemp1 > htemp2 ? htemp1 : htemp2
                 : htemp1 < htemp2 ? htemp1
                                   : htemp2);
            xnew = x + h;

            QUARISMA_CHECK(
                xnew != x,
                "Step size underflow (",
                h,
                " at tmp = ",
                x,
                ") in AdaptiveRungeKutta::rkqs");
            continue;
        }

        hnext = (errmax > adaptiverk_errcon)
                    ? adaptiverk_safety * h * std::pow(errmax, adaptiverk_pgrow)
                    : 5.0 * h;

        x += (hdid = h);
        if (!std::isnan(ytemp))
        {
            y = ytemp;
        }

        break;
    }
}

//-----------------------------------------------------------------------------
double runge_kutta_solver::solve_adaptive(
    const ODE_function& ode,
    const double        y_initial_condition,
    const double        x_initial_condition,
    const double        x,
    const double        tolerance,
    const double        step,
    const double        step_min,
    const size_t        adaptiverk_max_iteration,
    const double        adaptiverk_tiny,
    const double        adaptiverk_safety,
    const double        adaptiverk_pgrow,
    const double        adaptiverk_pshrink,
    const double        adaptiverk_errcon)
{
    double y   = y_initial_condition;
    double h   = (x_initial_condition <= x ? step : -step);
    double tmp = x_initial_condition;

    double hnext;
    double hdid;

    for (size_t i = 0; i < adaptiverk_max_iteration; ++i)
    {
        const auto dydx   = ode(tmp, y);
        const auto yScale = std::fabs(y) + std::fabs(dydx * h) + adaptiverk_tiny;

        if ((tmp + h - x) * (tmp + h - x_initial_condition) > 0.0)
        {
            h = x - tmp;
        }

        rkqs(
            y,
            tmp,
            hdid,
            hnext,
            dydx,
            h,
            tolerance,
            yScale,
            ode,
            adaptiverk_safety,
            adaptiverk_pgrow,
            adaptiverk_pshrink,
            adaptiverk_errcon);

        if ((tmp - x) * (x - x_initial_condition) >= 0.0)
        {
            return y;
        }

        QUARISMA_CHECK(
            std::fabs(hnext) > step_min,
            "Step size (",
            hnext,
            ") too small (",
            step_min,
            " min) in AdaptiveRungeKutta");

        h = hnext;
    }
    QUARISMA_THROW("Too many sttolerance (", adaptiverk_max_iteration, ") in AdaptiveRungeKutta");
}

//-----------------------------------------------------------------------------
double runge_kutta_solver::solve_fourth_order(
    const ODE_function& func,
    const double        y_initial_condition,
    const double        x_initial_condition,
    const double        x)
{
    const double step = x - x_initial_condition;
    const double k1   = step * func(x_initial_condition, y_initial_condition);
    const double k2 = step * func(x_initial_condition + 0.5 * step, y_initial_condition + 0.5 * k1);
    const double k3 = step * func(x_initial_condition + 0.5 * step, y_initial_condition + 0.5 * k2);
    const double k4 = step * func(x_initial_condition + step, y_initial_condition + k3);
    return y_initial_condition + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

}  // namespace quarisma
