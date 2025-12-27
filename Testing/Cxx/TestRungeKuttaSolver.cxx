#include <cmath>  // for exp, fabs

#include "common/macros.h"              // for quarisma
#include "common/runge_kutta_solver.h"  // for runge_kutta_solver
#include "util/logger.h"                // for QUARISMA_LOGF
#include "quarismaTest.h"                 // for ASSERT_ANY_THROW, EXPECT...

using namespace quarisma;

QUARISMATEST(Math, RungeKuttaSolver)
{
    START_LOG_TO_FILE_NAME(RungeKuttaSolver);
    // y=x^2exp(x)
    // y'=y+2xexp(x)
    const auto f = [](double x, double y) { return y + 2 * x * exp(x); };

    double x = 0.2;
    double y = x * x * exp(x);

    ASSERT_ANY_THROW(
        { runge_kutta_solver::solve_adaptive(f, 0.0, 0.0, x, 1.0e-6, 1.0e-4, 1., 10000); };);
    ASSERT_ANY_THROW(
        { runge_kutta_solver::solve_adaptive(f, 0.0, 0.0, x, 1.0e-6, 1.0e-4, 0., 10); };);

    double y1 = runge_kutta_solver::solve_adaptive(f, 0.0, 0.0, x);
    double y2 = runge_kutta_solver::solve_fourth_order(f, 0.0, 0.0, x);

    QUARISMA_LOGF(INFO, "expected:                 %f  ", y);
    QUARISMA_LOGF(INFO, "solve adaptive error:     %f  ", y1);
    QUARISMA_LOGF(INFO, "solve fourth order error: %f  ", y2);

    EXPECT_TRUE(std::fabs(y1 - y) < 0.00001);
    EXPECT_TRUE(std::fabs(y2 - y) < 0.00001);

    END_LOG_TO_FILE_NAME(RungeKuttaSolver);
    END_TEST();
}
