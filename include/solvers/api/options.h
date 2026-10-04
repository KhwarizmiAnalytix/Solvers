#ifndef SOLVERS_OPTIONS_H_
#define SOLVERS_OPTIONS_H_

#include <limits>
#include <optional>

#include "detail/support.h"
#include "solvers/api/backend_options.h"
#include "solvers/api/problem.h"
#include "solvers/api/status.h"

namespace solverslib::api
{
// User-facing knobs for solve(). Every field has an initialized default and is
// validated at the solve boundary (review F02). `algorithm` and `backend`
// default to automatic; the dispatcher resolves both from the problem traits,
// and either can be pinned to override the automatic choice.
struct solve_options
{
    api::algorithm algorithm = api::algorithm::automatic;
    api::backend   backend   = api::backend::automatic;

    // How to compute or select derivatives.
    api::derivative_mode derivatives = api::derivative_mode::automatic;

    // Separate budgets rather than one overloaded "iterations" count.
    int max_iterations = 100;
    // 0 keeps the backend default. Positive budgets are currently rejected at
    // the API boundary until all adapters can account for line-search and
    // finite-difference evaluations consistently.
    int max_function_evaluations = 0;

    // Named tolerances; units documented per quantity.
    double function_tolerance  = std::numeric_limits<double>::epsilon();
    double gradient_tolerance  = 0.0;
    double parameter_tolerance = std::numeric_limits<double>::epsilon();

    bool verbose = false;

    api::dispatch_policy policy;

    // Optional backend-specific tuning. Consulted only when the corresponding
    // backend is chosen; ignored otherwise. Absent means "backend defaults".
    std::optional<api::lm_options>        lm;
    std::optional<api::rnc_lm_options>    rnc_lm;
    std::optional<api::ceres_options>     ceres;
    std::optional<api::ipopt_options>     ipopt;
    std::optional<api::petsc_tao_options> petsc_tao;
};
}  // namespace solverslib::api

#endif  // SOLVERS_OPTIONS_H_
