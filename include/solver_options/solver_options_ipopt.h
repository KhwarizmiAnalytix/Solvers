#pragma once

#include <limits>
#include <memory>
#include <string>

#include "include/detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
// Ipopt is an interior-point solver for large-scale constrained nonlinear
// programming. These options cover the knobs the adapter forwards to Ipopt's
// option list; the third-party Ipopt types stay entirely inside the adapter
// (redesign review section 10).
enum class ipopt_hessian_approximation_enum : int
{
    EXACT           = 0,  // caller supplies a Hessian (or Hessian-vector) callback
    LIMITED_MEMORY  = 1   // Ipopt's L-BFGS quasi-Newton approximation
};

class SOLVER_VISIBILITY solver_options_ipopt : public solver_options
{
    friend class solver_options_ipopt_builder;

public:
    solver_options_ipopt(int max_num_iterations,
        double               function_tolerance  = 1e-8,
        double               gradient_tolerance  = 1e-8,
        double               parameter_tolerance = 1e-8,
        bool                 verbose             = false)
        : solver_options(solver_enum::IPOPT,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose)
    {
    }

    // Convergence tolerance on the scaled NLP error (Ipopt's "tol").
    double tol() const { return tol_; }
    void   set_tol(double value) { tol_ = value; }

    // Acceptable (relaxed) tolerance Ipopt may stop at after several iterations.
    double acceptable_tol() const { return acceptable_tol_; }
    void   set_acceptable_tol(double value) { acceptable_tol_ = value; }

    ipopt_hessian_approximation_enum hessian_approximation() const
    {
        return hessian_approximation_;
    }
    void set_hessian_approximation(ipopt_hessian_approximation_enum value)
    {
        hessian_approximation_ = value;
    }

    // Ipopt linear solver name (e.g. "mumps", "ma27"); empty keeps Ipopt's default.
    const std::string& linear_solver() const { return linear_solver_; }
    void               set_linear_solver(const std::string& value) { linear_solver_ = value; }

    double max_wall_time_seconds() const { return max_wall_time_seconds_; }
    void   set_max_wall_time_seconds(double value) { max_wall_time_seconds_ = value; }

    void set_log_file(const std::string& log_file) { log_file_ = log_file; }

private:
    solver_options_ipopt() : solver_options(solverslib::solver_enum::IPOPT) {}
    void initialize() const {}

    double                           tol_                   = 1e-8;
    double                           acceptable_tol_        = 1e-6;
    ipopt_hessian_approximation_enum hessian_approximation_ = ipopt_hessian_approximation_enum::LIMITED_MEMORY;
    std::string                      linear_solver_         = "";
    double                           max_wall_time_seconds_ = 1e9;
};

class SOLVER_VISIBILITY solver_options_ipopt_builder
{
public:
    solver_options_ipopt_builder()
        : options_(std::shared_ptr<solver_options_ipopt>(new solver_options_ipopt()))
    {
    }

    solver_options_ipopt_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }
    solver_options_ipopt_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }
    solver_options_ipopt_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }
    solver_options_ipopt_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }
    solver_options_ipopt_builder& with_verbose(bool val = true)
    {
        options_->set_debug(val);
        return *this;
    }
    solver_options_ipopt_builder& with_tol(double val)
    {
        options_->set_tol(val);
        return *this;
    }
    solver_options_ipopt_builder& with_acceptable_tol(double val)
    {
        options_->set_acceptable_tol(val);
        return *this;
    }
    solver_options_ipopt_builder& with_hessian_approximation(ipopt_hessian_approximation_enum val)
    {
        options_->set_hessian_approximation(val);
        return *this;
    }
    solver_options_ipopt_builder& with_linear_solver(const std::string& val)
    {
        options_->set_linear_solver(val);
        return *this;
    }
    solver_options_ipopt_builder& with_max_wall_time_seconds(double val)
    {
        options_->set_max_wall_time_seconds(val);
        return *this;
    }
    solver_options_ipopt_builder& with_log_file(const std::string& val)
    {
        options_->set_log_file(val);
        return *this;
    }

    std::shared_ptr<const solver_options_ipopt> build() const { return options_; }

private:
    std::shared_ptr<solver_options_ipopt> options_;
};
}  // namespace solverslib
