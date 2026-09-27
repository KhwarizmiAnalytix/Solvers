#include "solver_options/solver_options_ipopt.h"

namespace solverslib
{
solver_options_ipopt::solver_options_ipopt() : solver_options(solver_enum::IPOPT) {}

double                           solver_options_ipopt::tol() const { return tol_; }
double                           solver_options_ipopt::acceptable_tol() const { return acceptable_tol_; }
ipopt_hessian_approximation_enum solver_options_ipopt::hessian_approximation() const
{
    return hessian_approximation_;
}
const std::string& solver_options_ipopt::linear_solver() const { return linear_solver_; }
double             solver_options_ipopt::max_wall_time_seconds() const { return max_wall_time_seconds_; }

solver_options_ipopt_builder::solver_options_ipopt_builder()
    : options_(std::shared_ptr<solver_options_ipopt>(new solver_options_ipopt()))
{
}

solver_options_ipopt_builder& solver_options_ipopt_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_tol(double val)
{
    options_->tol_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_acceptable_tol(double val)
{
    options_->acceptable_tol_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_hessian_approximation(
    ipopt_hessian_approximation_enum val)
{
    options_->hessian_approximation_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_linear_solver(
    const std::string& val)
{
    options_->linear_solver_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_max_wall_time_seconds(double val)
{
    options_->max_wall_time_seconds_ = val;
    return *this;
}
solver_options_ipopt_builder& solver_options_ipopt_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}

std::shared_ptr<const solver_options_ipopt> solver_options_ipopt_builder::build() const
{
    return options_;
}
}  // namespace solverslib
