#pragma once

#include "api/dispatch.h"
#include "detail/api_access.h"
#include "solver_options/solver_options_rnc_lm.h"

namespace solverslib::api
{
// Native RNC-LM kernel. Only api::solve may call it.
class rnc_lm_solver
{
    SOLVERS_FRIEND_API_SOLVE

    static solver_result run(const least_squares_problem& problem,
        const vector_type&                                initial_guess,
        const solver_options_rnc_lm&                      options);
};
}  // namespace solverslib::api
