#include "solver_options/solver_options_bfgs.h"

#include "detail/support.h"

namespace solverslib
{
solver_options_bfgs::solver_options_bfgs() : solver_options(solver_enum::LBFGS) {}

void solver_options_bfgs::validate() const
{
    SOLVERS_CHECK(
        linesearch_tolerance_ > 0 && linesearch_tolerance_ < 0.5,
        "'linesearch_tolerance_' must satisfy 0 < linesearch_tolerance_ < 0.5");

    SOLVERS_CHECK(
        linesearch_wolfe_ > linesearch_tolerance_ && linesearch_wolfe_ < 1.,
        "'linesearch_wolfe_' must satisfy linesearch_tolerance_ < linesearch_wolfe_ < 1");
}

lbfgs_line_search_method_type solver_options_bfgs::method_type() const { return method_type_; }
lbfgs_line_search_type        solver_options_bfgs::type() const { return type_; }
size_t                        solver_options_bfgs::tau() const { return tau_; }
size_t solver_options_bfgs::max_iteration_linesearch() const { return max_iteration_linesearch_; }
double solver_options_bfgs::step_min() const { return step_min_; }
double solver_options_bfgs::step_max() const { return step_max_; }
double solver_options_bfgs::linesearch_tolerance() const { return linesearch_tolerance_; }
double solver_options_bfgs::linesearch_wolfe() const { return linesearch_wolfe_; }
double solver_options_bfgs::bump() const { return bump_; }

solver_options_bfgs_builder::solver_options_bfgs_builder()
    : options_(std::shared_ptr<solver_options_bfgs>(new solver_options_bfgs()))
{
}

solver_options_bfgs_builder& solver_options_bfgs_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_method_type(
    lbfgs_line_search_method_type val)
{
    options_->method_type_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_type(lbfgs_line_search_type val)
{
    options_->type_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_tau(size_t val)
{
    options_->tau_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_max_iteration_linesearch(size_t val)
{
    options_->max_iteration_linesearch_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_step_min(double val)
{
    options_->step_min_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_step_max(double val)
{
    options_->step_max_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_linesearch_tolerance(double val)
{
    options_->linesearch_tolerance_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_linesearch_wolfe(double val)
{
    options_->linesearch_wolfe_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_bump(double val)
{
    options_->bump_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}
solver_options_bfgs_builder& solver_options_bfgs_builder::with_aad_jacobian(bool val)
{
    options_->aad_jacobian_ = val;
    return *this;
}

std::shared_ptr<const solver_options_bfgs> solver_options_bfgs_builder::build() const
{
    return options_;
}
}  // namespace solverslib
