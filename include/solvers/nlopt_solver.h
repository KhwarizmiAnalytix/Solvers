#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "detail/api_access.h"
#include "detail/backend_status.h"
#include "detail/eigen_support.h"
#include "detail/support.h"

namespace solverslib
{
class solver_options_nlopt;

// Adapter around NLopt's optimization solver for general (optionally
// bound-constrained) scalar-objective optimization. NLopt is a library for
// nonlinear local and global optimization. This adapter exposes NLopt's
// capabilities through std::function callbacks and std::vector<double>,
// mirroring ceres_solver and ipopt_solver. The implementation is compiled
// against NLopt only when SOLVERS_HAS_NLOPT is set; otherwise is_supported()
// is false and solve() is a no-op the dispatcher never reaches.
class nlopt_solver
{
    SOLVERS_FRIEND_API_SOLVE

public:
    using objective_type = std::function<double(const vector_type&)>;
    using gradient_type  = std::function<void(const vector_type&, vector_type&)>;

    SOLVER_API static bool is_supported();

private:
    SOLVER_API nlopt_solver(size_t num_parameters,
        objective_type             objective,
        gradient_type              gradient,
        std::vector<double>        lower_bounds = {},
        std::vector<double>        upper_bounds = {});

    // Reports how the solve ended (budget vs. failure vs. infeasible) and
    // the raw NLopt result code in native_code.
    SOLVER_API backend_solve_status solve(
        std::vector<double>& parameters, const solver_options_nlopt& options);

    size_t              num_parameters_;
    objective_type      objective_;
    gradient_type       gradient_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};
}  // namespace solverslib
