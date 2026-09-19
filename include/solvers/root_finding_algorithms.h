#pragma once

#include <cstddef>
#include <functional>
#include <limits>

#include "include/detail/support.h"
#include "solver_options/root_finding_options.h"

namespace solverslib
{
/**
 * @brief Scalar (1-D) root-finding algorithms, all configured through a
 * single root_finding_options instance rather than per-call trailing
 * parameter lists.
 *
 * Bracketing methods (bisection, false_position, ridders, dekker, brent)
 * require x1/x2 to bracket a sign change and are guaranteed to converge.
 * Open methods (newton_raphson, secant) only need one or two starting
 * points and converge faster when they converge, but are not guaranteed to.
 */
class root_finding_algorithms
{
public:
    using function_type          = std::function<double(double)>;
    using function_gradient_type = std::function<double(double, double&)>;

    // -- Bracketing methods --------------------------------------------------

    /**
     * @brief Simple bisection: halves the bracket every iteration. The
     * slowest bracketing method (linear convergence) but the most robust -
     * useful as a fallback when a function's derivative is unreliable.
     */
    SOLVER_API static bool bisection(function_type const& func,
        double                                            x1,
        double                                            x2,
        double&                                           root,
        const root_finding_options&                       options = root_finding_options());

    /**
     * @brief Regula falsi (false position): like bisection but replaces the
     * midpoint with the linear-interpolation root of the secant line
     * through the bracket endpoints. Uses the Illinois modification (damping
     * a stagnant endpoint) to avoid the classic slow one-sided convergence
     * of plain false position.
     */
    SOLVER_API static bool false_position(function_type const& func,
        double                                                x1,
        double                                                x2,
        double&                                               root,
        const root_finding_options&                           options = root_finding_options());

    /**
     * @brief Ridders' method: combines a bisection step with exponential
     * (Ridders) interpolation for quadratic-ish convergence without needing
     * a derivative.
     */
    SOLVER_API static bool ridders(function_type const& func,
        double                                          x1,
        double                                          x2,
        double&                                         root,
        const root_finding_options&                     options = root_finding_options());

    /**
     * @brief Dekker's method: hybrid secant/bisection using the function's
     * derivative to choose the next iterate, falling back to bisection when
     * the secant step would leave the bracket. Predecessor of brent().
     */
    SOLVER_API static bool dekker(function_gradient_type const& func,
        double                                                  x1,
        double                                                  x2,
        double&                                                 result,
        const root_finding_options&                             options = root_finding_options());

    /**
     * @brief Brent's method: Dekker's method refined with inverse quadratic
     * interpolation and stricter step-acceptance rules. The most robust
     * general-purpose bracketing solver here; prefer it when unsure.
     */
    SOLVER_API static bool brent(function_type const& func,
        double                                        x1,
        double                                        x2,
        double&                                        root,
        const root_finding_options&                    options = root_finding_options());

    // -- Open (non-bracketing) methods ---------------------------------------

    /**
     * @brief Newton-Raphson: quadratic convergence from a single starting
     * point using the function's derivative, but not guaranteed to converge
     * (can diverge or cycle if the derivative is small or the guess poor).
     */
    SOLVER_API static bool newton_raphson(function_gradient_type const& func,
        double                                                         x0,
        double&                                                       root,
        const root_finding_options&                                   options =
            root_finding_options());

    /**
     * @brief Secant method: Newton-Raphson's derivative-free counterpart,
     * approximating the derivative from two starting points.
     */
    SOLVER_API static bool secant(function_type const& func,
        double                                         x0,
        double                                         x1,
        double&                                        root,
        const root_finding_options&                    options = root_finding_options());

private:
    SOLVERS_DELETE_CLASS(root_finding_algorithms);
};
}  // namespace solverslib
