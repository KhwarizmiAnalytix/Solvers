#pragma once

#include "api/dispatch.h"
#include "api/result.h"
#include "solver_options/solver_options_rnc_lm.h"

namespace solverslib::api
{
// Native RNC-LM entry point. Requires curve derivatives in the problem.
SOLVER_API solver_result solve_rnc_lm(
    const least_squares_problem&     problem,
    const vector_type&               initial_guess,
    const solver_options_rnc_lm&     options);
}  // namespace solverslib::api
