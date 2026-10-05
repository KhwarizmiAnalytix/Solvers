#pragma once

#include "solver_options/solver_options.h"
#include "solvers/api/result.h"
#include "solvers/api/solve.h"

namespace solverslib::api
{
// Native RNC-LM entry point. Requires curve derivatives in the problem.
SOLVER_API solver_result solve_rnc_lm(
    const least_squares_problem& problem,
    const vector_type&           initial_guess,
    const solve_options&         options);
}  // namespace solverslib::api
