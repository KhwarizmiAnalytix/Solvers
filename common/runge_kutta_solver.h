#pragma once

#include <cstddef>
#include <functional>

#include "MathModule.h"
#include "common/macros.h"

namespace quarisma
{
class runge_kutta_solver
{
public:
    using ODE_function = std::function<double(const double, const double)>;

    //Runge Kutta Fehlberg method
    MATH_API static double solve_adaptive(
        const ODE_function& ode,
        const double        y_initial_condition,
        const double        x_initial_condition,
        const double        x,
        const double        tolerance                = 1.0e-6,
        const double        step                     = 1.0e-4,
        const double        step_min                 = 0.,
        const size_t        adaptiverk_max_iteration = 1000,
        const double        adaptiverk_tiny          = 1.0e-30,
        const double        adaptiverk_safety        = 0.9,
        const double        adaptiverk_pgrow         = -0.2,
        const double        adaptiverk_pshrink       = -0.25,
        const double        adaptiverk_errcon        = 1.89e-4);

    MATH_API static double solve_fourth_order(
        const ODE_function& func,
        const double        y_initial_condition,
        const double        x_initial_condition,
        const double        x);

private:
    QUARISMA_DELETE_CLASS(runge_kutta_solver);
};
}  // namespace quarisma
