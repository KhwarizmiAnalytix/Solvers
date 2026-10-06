#ifndef SOLVERS_ROOT_FINDING_ALGORITHMS_H_
#define SOLVERS_ROOT_FINDING_ALGORITHMS_H_

#include <cstddef>
#include <functional>

#include "solver_options/root_finding_options.h"

// Root-finding iterations with diagnostics. The public run_xxx functions
// are bool/out-parameter wrappers over these.
namespace solverslib::detail
{
enum class root_outcome
{
    converged,        // a tolerance was met; `root` is the answer
    iteration_limit,  // budget used up; `root` is the best estimate so far
    degenerate        // the method could not continue (e.g. zero interpolation denominator)
};

struct root_run
{
    root_outcome outcome    = root_outcome::iteration_limit;
    double       root       = 0.0;
    std::size_t  iterations = 0;
};

using scalar_function          = std::function<double(double)>;
using scalar_function_gradient = std::function<double(double, double&)>;

// Bracketing methods require f(x1) * f(x2) <= 0 and throw otherwise (the
// structured entry point checks the bracket first and reports it instead).
root_run run_bisection(const scalar_function& f, double x1, double x2, const root_finding_options&);
root_run run_false_position(
    const scalar_function& f, double x1, double x2, const root_finding_options&);
root_run run_ridders(const scalar_function& f, double x1, double x2, const root_finding_options&);
root_run run_brent(const scalar_function& f, double x1, double x2, const root_finding_options&);
root_run run_dekker(
    const scalar_function_gradient& f, double x1, double x2, const root_finding_options&);

// Open methods. A vanishing derivative or secant denominator throws.
root_run run_newton_raphson(
    const scalar_function_gradient& f, double x0, const root_finding_options&);
root_run run_secant(const scalar_function& f, double x0, double x1, const root_finding_options&);
}  // namespace solverslib::detail

#endif  // SOLVERS_ROOT_FINDING_ALGORITHMS_H_
