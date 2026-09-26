#pragma once

#include "include/detail/support.h"
#include "solvers/api/options.h"
#include "solvers/api/problem.h"
#include "solvers/api/result.h"

namespace solverslib::api
{
// -- trait derivation --------------------------------------------------------
problem_traits inspect(const least_squares_problem& problem);
problem_traits inspect(const optimization_problem& problem);

// A problem is "large scale" when it exceeds a size threshold or when the
// caller supplied matrix-free products the dispatcher should prefer to exploit
// (review section 7).
bool is_large_scale(const problem_traits& traits, const dispatch_policy& policy);

// Resolve backend::automatic / algorithm::automatic to concrete choices from
// the traits, honoring any explicit pin in `options`. Exposed for testing the
// dispatch rules independently of execution.
backend   select_backend(const problem_traits& traits, const solve_options& options);
algorithm select_algorithm(const problem_traits& traits, const solve_options& options);

// -- entry points ------------------------------------------------------------
// Inspect traits, validate, choose a backend/algorithm, run it, and return a
// structured result. The initial iterate is taken by value and never mutated;
// the accepted iterate is returned in solver_result::parameters instead of
// being written back to caller storage (review: invariant 2, and "The new API
// accepts an initial point and returns a result").
solver_result solve(const least_squares_problem& problem,
    const vector_type&                            initial_guess,
    const solve_options&                          options = {});

solver_result solve(const optimization_problem& problem,
    const vector_type&                          initial_guess,
    const solve_options&                        options = {});
}  // namespace solverslib::api
