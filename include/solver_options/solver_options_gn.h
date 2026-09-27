#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "include/detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
/**
 * @brief Options for the plain (undamped) Gauss-Newton solver.
 *
 * Unlike Levenberg-Marquardt, Gauss-Newton has no trust-region/damping
 * parameter to keep the step in check when the linearized model is a poor
 * fit; instead it globalizes convergence with an Armijo backtracking line
 * search along the (guaranteed-descent) Gauss-Newton direction.
 */
class SOLVER_VISIBILITY solver_options_gn : public solver_options
{
    friend class solver_options_gn_builder;

public:
    solver_options_gn(int max_num_iterations,
        double                function_tolerance  = std::numeric_limits<double>::epsilon(),
        double                gradient_tolerance  = 0.0,
        double                parameter_tolerance = std::numeric_limits<double>::epsilon(),
        bool                  verbose             = false)
        : solver_options(solver_enum::GAUSS_NEWTON,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose)
    {
    }

    // Getters and setters
    double bump() const { return bump_; }
    void   set_bump(double bump) { bump_ = bump; }

    size_t max_line_search_iterations() const { return max_line_search_iterations_; }
    void   set_max_line_search_iterations(size_t value) { max_line_search_iterations_ = value; }

    double line_search_backtracking_factor() const { return line_search_backtracking_factor_; }
    void   set_line_search_backtracking_factor(double value)
    {
        line_search_backtracking_factor_ = value;
    }

    double line_search_sufficient_decrease() const { return line_search_sufficient_decrease_; }
    void   set_line_search_sufficient_decrease(double value)
    {
        line_search_sufficient_decrease_ = value;
    }

    void set_log_file(const std::string& log_file) { log_file_ = log_file; }

private:
    solver_options_gn() : solver_options(solverslib::solver_enum::GAUSS_NEWTON) {};

    void initialize() const {};

    double bump_                            = 0.00001;
    size_t max_line_search_iterations_      = 20;
    double line_search_backtracking_factor_ = 0.5;
    double line_search_sufficient_decrease_ = 1e-4;
};

/**
 * @brief Builder for solver_options_gn configuration
 *
 * Usage:
 *   auto options = solver_options_gn_builder()
 *       .with_max_iterations(100).with_function_tolerance(1e-6).build();
 */
class SOLVER_VISIBILITY solver_options_gn_builder
{
public:
    solver_options_gn_builder()
        : options_(std::shared_ptr<solver_options_gn>(new solver_options_gn()))
    {
    }

    // Base solver options
    solverslib::solver_options_gn_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_verbose(bool val = true)
    {
        options_->set_debug(val);
        return *this;
    }

    // Gauss-Newton-specific options
    solverslib::solver_options_gn_builder& with_bump(double val)
    {
        options_->set_bump(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_max_line_search_iterations(size_t val)
    {
        options_->set_max_line_search_iterations(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_line_search_backtracking_factor(double val)
    {
        options_->set_line_search_backtracking_factor(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_line_search_sufficient_decrease(double val)
    {
        options_->set_line_search_sufficient_decrease(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_log_file(const std::string& val)
    {
        options_->set_log_file(val);
        return *this;
    }

    solverslib::solver_options_gn_builder& with_aad_jacobian(bool val = true)
    {
        options_->aad_jacobian_ = val;
        return *this;
    }

    // Build the final options
    std::shared_ptr<const solver_options_gn> build() const { return options_; }

private:
    std::shared_ptr<solver_options_gn> options_;
};

}  // namespace solverslib
