#pragma once
#include "solvers/api/options.h"
#include "solvers/api/result.h"

namespace solverslib
{
// Native RNC-LM entry point. Also dispatched by api::solve with
// algorithm::riemann_normal_coordinate_lm. Requires curve derivatives.
SOLVER_API api::solver_result solve_rnc_lm(const api::least_squares_problem& problem,
    const vector_type&                                                       initial_guess,
    const api::solve_options&                                                options);
}  // namespace solverslib
