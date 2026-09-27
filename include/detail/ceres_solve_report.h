#ifndef SOLVERS_CERES_SOLVE_REPORT_H_
#define SOLVERS_CERES_SOLVE_REPORT_H_

#if SOLVERS_HAS_CERES

#include <ceres/ceres.h>

#include "solvers/api/status.h"

namespace solverslib::detail
{

// Internal report capturing Ceres Solver::Summary details
struct ceres_solve_report
{
    // Termination information
    ceres::TerminationType termination_type = ceres::TerminationType::NO_CONVERGENCE;
    ceres::Convergence convergence_reason = ceres::Convergence::CONVERGENCE_FAILURE;
    bool is_solution_usable = false;

    // Iteration statistics
    int iterations = 0;
    int inner_iterations = 0;
    int line_search_iterations = 0;

    // Timing
    double total_time_seconds = 0.0;
    double solve_time_seconds = 0.0;
    double setup_time_seconds = 0.0;

    // Evaluation counts
    int residual_evaluations = 0;
    int jacobian_evaluations = 0;
    int linear_solver_runs = 0;

    // Translate to API status
    api::solver_status to_api_status() const
    {
        if (termination_type == ceres::CONVERGENCE)
        {
            return api::solver_status::converged;
        }
        if (termination_type == ceres::NO_CONVERGENCE)
        {
            // Check if we have a usable iterate (hit iteration limit)
            if (is_solution_usable && iterations > 0)
            {
                return api::solver_status::max_iterations;
            }
            return api::solver_status::numerical_failure;
        }
        // FAILURE, USER_FAILURE, etc.
        return api::solver_status::numerical_failure;
    }

    // Extract from Ceres summary
    static ceres_solve_report from_ceres_summary(const ceres::Solver::Summary& summary)
    {
        ceres_solve_report report;
        report.termination_type = summary.termination_type;
        report.is_solution_usable = summary.IsSolutionUsable();
        report.iterations = summary.iterations.size();
        report.inner_iterations = summary.num_inner_iteration_steps;
        report.line_search_iterations = summary.num_line_search_direction_restarts;
        report.total_time_seconds = summary.total_time_in_seconds;
        report.solve_time_seconds = summary.solve_time_in_seconds;
        report.setup_time_seconds = summary.setup_time_in_seconds;
        report.residual_evaluations = summary.num_residual_evaluations;
        report.jacobian_evaluations = summary.num_jacobian_evaluations;
        report.linear_solver_runs = summary.num_linear_solver_runs;
        return report;
    }
};

}  // namespace solverslib::detail

#endif  // SOLVERS_HAS_CERES

#endif  // SOLVERS_CERES_SOLVE_REPORT_H_
