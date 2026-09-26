#pragma once

#include <limits>
#include <optional>

#include "include/detail/support.h"
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

    // Separate budgets rather than one overloaded "iterations" count.
    int max_iterations           = 100;
    int max_function_evaluations = 0;  // 0 == unlimited

    // Named tolerances; units documented per quantity.
    double function_tolerance  = std::numeric_limits<double>::epsilon();
    double gradient_tolerance  = 0.0;
    double parameter_tolerance = std::numeric_limits<double>::epsilon();

    bool verbose = false;

    api::dispatch_policy policy;

    // Optional backend-specific tuning. Consulted only when the corresponding
    // backend is chosen; ignored otherwise. Absent means "backend defaults".
    std::optional<api::ceres_options>     ceres;
    std::optional<api::nlopt_options>     nlopt;
    std::optional<api::ipopt_options>     ipopt;
    std::optional<api::petsc_tao_options> petsc_tao;
};
}  // namespace solverslib::api
