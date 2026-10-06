#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "detail/support.h"

namespace solverslib
{
// Single-double convenience forms. They return one root chosen by a fixed rule
// (and, for the quartic, clamp it to a threshold) and report no failure.
class polynomial_solver
{
    SOLVERS_DELETE_CLASS(polynomial_solver);

public:
    //-----------------------------------------------------------------------------
    /*
    //tex:
    //fourth degree polynomial solve of the equation: $$x^4 + a_3x^3 + a_2x^2 + a_1x + a_0=0$$
    // Returns the LARGEST real root, raised to `threshold` if it is smaller. (This
    // comment used to say "minimum positive root"; the implementation has always
    // returned the largest.) Equivalent to
    // api::at_least(api::largest_root(api::real_roots_quartic(...).roots), threshold)
    // when a real root exists.
    */
    SOLVER_API static double fourth_degree_polynomial_solver(
        double a3, double a2, double a1, double a0, double threshold = 0.0000);

    //-----------------------------------------------------------------------------
    /*
    //tex:
    //cubic solver of the equation: $$x^3+bx^2+cx+d=0$$
    */
    SOLVER_API static double third_degree_polynomial_solver(double b, double c, double d);

    //------------------------------------------------------------------------------
    /*
    //tex:
    //quadratic solver of the equation: $$x^2+bx+c=0$$
    //return the minimum positive root.
    */
    SOLVERS_FORCE_INLINE static double second_degree_polynomial_solver(double b, double c) noexcept
    {
        const auto b2 = b * b;

        auto delta = b2 - 4. * c;
        if (delta > 0.)
        {
            delta = sqrt(delta);

            const auto root_max = (-b + delta) * 0.5;
            const auto root_min = (-b - delta) * 0.5;

            return root_min > 0. ? root_min : root_max;
        }

        return std::numeric_limits<double>::signaling_NaN();
    }
};
}  // namespace solverslib
