#ifndef SOLVERS_SOLVE_H_
#define SOLVERS_SOLVE_H_

#include "detail/support.h"
#include "solver_options/solver_options.h"
#include "solvers/api/problem.h"
#include "solvers/api/result.h"

namespace solverslib::api
{
// -- trait derivation --------------------------------------------------------
problem_traits inspect(const least_squares_problem& problem);
problem_traits inspect(const optimization_problem& problem);

// -- entry points ------------------------------------------------------------
// Inspect traits, validate, choose a backend/algorithm, run it, and return a
// structured result. The initial iterate is taken by value and never mutated;
// the accepted iterate is returned in solver_result::parameters instead of
// being written back to caller storage (review: invariant 2, and "The new API
// accepts an initial point and returns a result").
solver_result solve(const least_squares_problem& problem,
    const vector_type&                           initial_guess,
    const ::solverslib::solve_options&           options = {});

solver_result solve(const optimization_problem& problem,
    const vector_type&                          initial_guess,
    const ::solverslib::solve_options&          options = {});
}  // namespace solverslib::api

#endif  // SOLVERS_SOLVE_H_
