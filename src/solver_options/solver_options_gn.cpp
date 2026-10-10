#include "solver_options/solver_options_gn.h"

namespace solverslib
{
solver_options_gn::solver_options_gn() : solver_options(solver_enum::GAUSS_NEWTON) {}

double solver_options_gn::bump() const { return bump_; }
finite_difference_scale solver_options_gn::difference_scale() const { return difference_scale_; }
size_t solver_options_gn::max_line_search_iterations() const { return max_line_search_iterations_; }
double solver_options_gn::line_search_backtracking_factor() const
{
    return line_search_backtracking_factor_;
}
double solver_options_gn::line_search_sufficient_decrease() const
{
    return line_search_sufficient_decrease_;
}

solver_options_gn_builder::solver_options_gn_builder()
    : options_(std::shared_ptr<solver_options_gn>(new solver_options_gn()))
{
}

solver_options_gn_builder& solver_options_gn_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_bump(double val)
{
    options_->bump_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_difference_scale(
    finite_difference_scale val)
{
    options_->difference_scale_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_max_line_search_iterations(size_t val)
{
    options_->max_line_search_iterations_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_line_search_backtracking_factor(
    double val)
{
    options_->line_search_backtracking_factor_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_line_search_sufficient_decrease(
    double val)
{
    options_->line_search_sufficient_decrease_ = val;
    return *this;
}
solver_options_gn_builder& solver_options_gn_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}

std::shared_ptr<const solver_options_gn> solver_options_gn_builder::build() const
{
    return options_;
}
}  // namespace solverslib
