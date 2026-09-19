#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "include/detail/support.h"
#include "include/detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class lbfgs_line_search_method_type : int
{
    ARMIJO       = 1,
    WOLFE        = 2,
    STRONG_WOLFE = 3
};

enum class lbfgs_line_search_type : int
{
    BACKTRACKING   = 1,
    NOCEDAL_WRIGHT = 2,
    BRACKETING     = 3
};

class SOLVER_VISIBILITY solver_options_bfgs : public solver_options
{
    friend class solver_options_bfgs_builder;

public:
    solver_options_bfgs(
        int    max_num_iterations,
        double function_tolerance  = std::numeric_limits<double>::epsilon(),
        double gradient_tolerance  = std::numeric_limits<double>::epsilon(),
        double parameter_tolerance = std::numeric_limits<double>::epsilon(),
        bool   verbose             = false)
        : solver_options(
              solver_enum::LBFGS,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose)
    {
        validate();
    }

    // Getters
    lbfgs_line_search_method_type method_type() const { return method_type_; }
    lbfgs_line_search_type        type() const { return type_; }
    size_t                        tau() const { return tau_; }
    size_t max_iteration_linesearch() const { return max_iteration_linesearch_; }
    double step_min() const { return step_min_; }
    double step_max() const { return step_max_; }
    double linesearch_tolerance() const { return linesearch_tolerance_; }
    double linesearch_wolfe() const { return linesearch_wolfe_; }
    double bump() const { return bump_; }

    // Setters
    void set_method_type(lbfgs_line_search_method_type method_type) { method_type_ = method_type; }
    void set_type(lbfgs_line_search_type type) { type_ = type; }
    void set_tau(size_t tau) { tau_ = tau; }
    void set_max_iteration_linesearch(size_t max_iteration_linesearch)
    {
        max_iteration_linesearch_ = max_iteration_linesearch;
    }
    void set_step_min(double step_min) { step_min_ = step_min; }
    void set_step_max(double step_max) { step_max_ = step_max; }
    void set_linesearch_tolerance(double linesearch_tolerance)
    {
        linesearch_tolerance_ = linesearch_tolerance;
    }
    void set_linesearch_wolfe(double linesearch_wolfe) { linesearch_wolfe_ = linesearch_wolfe; }
    void set_bump(double bump) { bump_ = bump; }

private:
    SOLVER_API void validate() const;

    void initialize() const { validate(); };

    solver_options_bfgs() : solver_options(solverslib::solver_enum::LBFGS) {};



    lbfgs_line_search_method_type method_type_ = solverslib::lbfgs_line_search_method_type::ARMIJO;
    lbfgs_line_search_type        type_        = lbfgs_line_search_type::NOCEDAL_WRIGHT;
    size_t                        tau_         = 6;
    size_t                        max_iteration_linesearch_ = 40;
    double                        step_min_                 = 0.;
    double                        step_max_                 = 1000000;
    double                        linesearch_tolerance_ = std::numeric_limits<double>::epsilon();
    double                        linesearch_wolfe_     = 0.9;
    double                        bump_                 = 0.000001;
};

/**
 * @brief Builder for solver_options_bfgs configuration
 *
 * Usage:
 *   auto options = solver_options_bfgs_builder()
 *       .with_max_iterations(100).with_function_tolerance(1e-6)
 *       .with_method_type(lbfgs_line_search_method_type::ARMIJO).with_tau(10).build();
 */
class SOLVER_VISIBILITY solver_options_bfgs_builder
{
public:
    SOLVER_API solver_options_bfgs_builder()
        : options_(std::shared_ptr<solver_options_bfgs>(new solver_options_bfgs()))
    {
    }

    // Base solver options
    solverslib::solver_options_bfgs_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_verbose(bool val = true)
    {
        options_->verbose_ = val;
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_log_file(const std::string& val)
    {
        options_->log_file_ = val;
        return *this;
    }

    // BFGS-specific options
    solverslib::solver_options_bfgs_builder& with_method_type(lbfgs_line_search_method_type val)
    {
        options_->set_method_type(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_type(lbfgs_line_search_type val)
    {
        options_->set_type(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_tau(size_t val)
    {
        options_->set_tau(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_max_iteration_linesearch(size_t val)
    {
        options_->set_max_iteration_linesearch(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_step_min(double val)
    {
        options_->set_step_min(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_step_max(double val)
    {
        options_->set_step_max(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_linesearch_tolerance(double val)
    {
        options_->set_linesearch_tolerance(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_linesearch_wolfe(double val)
    {
        options_->set_linesearch_wolfe(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_bump(double val)
    {
        options_->set_bump(val);
        return *this;
    }

    solverslib::solver_options_bfgs_builder& with_aad_jacobian(bool val = true)
    {
        options_->aad_jacobian_ = val;
        return *this;
    }

    // Build the final options
    SOLVER_API std::shared_ptr<const solver_options_bfgs> build() const { return options_; }

private:
    std::shared_ptr<solver_options_bfgs> options_;
};

}  // namespace solverslib
