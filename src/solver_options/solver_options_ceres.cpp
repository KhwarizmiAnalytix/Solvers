#include "solver_options/solver_options_ceres.h"

namespace solverslib
{
solver_options_ceres::solver_options_ceres() : solver_options(solver_enum::CERES) {}

visibility_clustering_enum solver_options_ceres::visibility_clustering_type() const
{
    return visibility_clustering_type_;
}
preconditioner_enum solver_options_ceres::preconditioner_type() const
{
    return preconditioner_type_;
}
sparse_linear_algebra_library_enum solver_options_ceres::sparse_linear_algebra_library_type() const
{
    return sparse_linear_algebra_library_type_;
}
dense_linear_algebra_library_enum solver_options_ceres::dense_linear_algebra_library_type() const
{
    return dense_linear_algebra_library_type_;
}
linear_solver_enum solver_options_ceres::linear_solver_type() const { return linear_solver_type_; }
linear_solver_ordering_enum solver_options_ceres::linear_solver_ordering_type() const
{
    return linear_solver_ordering_type_;
}
dogleg_enum solver_options_ceres::dogleg_type() const { return dogleg_type_; }
trust_region_strategy_enum solver_options_ceres::trust_region_strategy_type() const
{
    return trust_region_strategy_type_;
}
line_search_interpolation_enum solver_options_ceres::line_search_interpolation_type() const
{
    return line_search_interpolation_type_;
}
minimizer_enum solver_options_ceres::minimizer_type() const { return minimizer_type_; }
line_search_direction_enum solver_options_ceres::line_search_direction_type() const
{
    return line_search_direction_type_;
}
line_search_enum solver_options_ceres::line_search_type() const { return line_search_type_; }
nonlinear_conjugate_gradient_enum solver_options_ceres::nonlinear_conjugate_gradient_type() const
{
    return nonlinear_conjugate_gradient_type_;
}
numeric_diff_method_enum solver_options_ceres::numeric_diff_method() const
{
    return numeric_diff_method_;
}
int    solver_options_ceres::max_lbfgs_rank() const { return max_lbfgs_rank_; }
double solver_options_ceres::initial_trust_region_radius() const
{
    return initial_trust_region_radius_;
}
double solver_options_ceres::max_trust_region_radius() const { return max_trust_region_radius_; }
double solver_options_ceres::min_trust_region_radius() const { return min_trust_region_radius_; }
double solver_options_ceres::min_relative_decrease() const { return min_relative_decrease_; }
double solver_options_ceres::eta() const { return eta_; }
bool   solver_options_ceres::use_inner_iterations() const { return use_inner_iterations_; }
double solver_options_ceres::inner_iteration_tolerance() const
{
    return inner_iteration_tolerance_;
}
bool   solver_options_ceres::jacobi_scaling() const { return jacobi_scaling_; }
bool   solver_options_ceres::dynamic_sparsity() const { return dynamic_sparsity_; }
bool   solver_options_ceres::use_mixed_precision_solves() const
{
    return use_mixed_precision_solves_;
}
int solver_options_ceres::max_num_refinement_iterations() const
{
    return max_num_refinement_iterations_;
}
double solver_options_ceres::min_line_search_step_size() const
{
    return min_line_search_step_size_;
}
double solver_options_ceres::line_search_sufficient_function_decrease() const
{
    return line_search_sufficient_function_decrease_;
}
double solver_options_ceres::line_search_sufficient_curvature_decrease() const
{
    return line_search_sufficient_curvature_decrease_;
}
double solver_options_ceres::max_line_search_step_contraction() const
{
    return max_line_search_step_contraction_;
}
double solver_options_ceres::min_line_search_step_contraction() const
{
    return min_line_search_step_contraction_;
}
double solver_options_ceres::max_line_search_step_expansion() const
{
    return max_line_search_step_expansion_;
}
double solver_options_ceres::gradient_check_relative_precision() const
{
    return gradient_check_relative_precision_;
}
double solver_options_ceres::gradient_check_numeric_derivative_relative_step_size() const
{
    return gradient_check_numeric_derivative_relative_step_size_;
}
double solver_options_ceres::max_solver_time_in_seconds() const
{
    return max_solver_time_in_seconds_;
}
int solver_options_ceres::num_threads() const { return num_threads_; }
int solver_options_ceres::min_linear_solver_iterations() const
{
    return min_linear_solver_iterations_;
}
int solver_options_ceres::max_linear_solver_iterations() const
{
    return max_linear_solver_iterations_;
}
const std::vector<int>& solver_options_ceres::trust_region_minimizer_iterations_to_dump() const
{
    return trust_region_minimizer_iterations_to_dump_;
}
const std::string& solver_options_ceres::trust_region_problem_dump_directory() const
{
    return trust_region_problem_dump_directory_;
}
dump_format_enum solver_options_ceres::trust_region_problem_dump_format_type() const
{
    return trust_region_problem_dump_format_type_;
}

// Builder
solver_options_ceres_builder::solver_options_ceres_builder()
    : options_(std::shared_ptr<solver_options_ceres>(new solver_options_ceres()))
{
}

solver_options_ceres_builder& solver_options_ceres_builder::with_max_iterations(int val)
{
    options_->max_num_iterations_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_function_tolerance(double val)
{
    options_->function_tolerance_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_gradient_tolerance(double val)
{
    options_->gradient_tolerance_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_parameter_tolerance(double val)
{
    options_->parameter_tolerance_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_verbose(bool val)
{
    options_->verbose_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_log_file(const std::string& val)
{
    options_->log_file_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_linear_solver_type(
    linear_solver_enum val)
{
    options_->linear_solver_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_preconditioner_type(
    preconditioner_enum val)
{
    options_->preconditioner_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_minimizer_type(minimizer_enum val)
{
    options_->minimizer_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_trust_region_strategy_type(
    trust_region_strategy_enum val)
{
    options_->trust_region_strategy_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_num_threads(int val)
{
    options_->num_threads_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_solver_time_in_seconds(
    double val)
{
    options_->max_solver_time_in_seconds_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_dogleg_type(dogleg_enum val)
{
    options_->dogleg_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_line_search_direction_type(
    line_search_direction_enum val)
{
    options_->line_search_direction_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_line_search_type(
    line_search_enum val)
{
    options_->line_search_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_nonlinear_conjugate_gradient_type(
    nonlinear_conjugate_gradient_enum val)
{
    options_->nonlinear_conjugate_gradient_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_line_search_interpolation_type(
    line_search_interpolation_enum val)
{
    options_->line_search_interpolation_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_numeric_diff_method(
    numeric_diff_method_enum val)
{
    options_->numeric_diff_method_ = val;
    return *this;
}
solver_options_ceres_builder&
solver_options_ceres_builder::with_sparse_linear_algebra_library_type(
    sparse_linear_algebra_library_enum val)
{
    options_->sparse_linear_algebra_library_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_dense_linear_algebra_library_type(
    dense_linear_algebra_library_enum val)
{
    options_->dense_linear_algebra_library_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_linear_solver_ordering_type(
    linear_solver_ordering_enum val)
{
    options_->linear_solver_ordering_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_visibility_clustering_type(
    visibility_clustering_enum val)
{
    options_->visibility_clustering_type_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_initial_trust_region_radius(
    double val)
{
    options_->initial_trust_region_radius_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_trust_region_radius(double val)
{
    options_->max_trust_region_radius_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_min_trust_region_radius(double val)
{
    options_->min_trust_region_radius_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_min_relative_decrease(double val)
{
    options_->min_relative_decrease_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_eta(double val)
{
    options_->eta_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_use_inner_iterations(bool val)
{
    options_->use_inner_iterations_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_inner_iteration_tolerance(
    double val)
{
    options_->inner_iteration_tolerance_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_jacobi_scaling(bool val)
{
    options_->jacobi_scaling_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_dynamic_sparsity(bool val)
{
    options_->dynamic_sparsity_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_use_mixed_precision_solves(
    bool val)
{
    options_->use_mixed_precision_solves_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_num_refinement_iterations(
    int val)
{
    options_->max_num_refinement_iterations_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_lbfgs_rank(int val)
{
    options_->max_lbfgs_rank_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_min_line_search_step_size(
    double val)
{
    options_->min_line_search_step_size_ = val;
    return *this;
}
solver_options_ceres_builder&
solver_options_ceres_builder::with_line_search_sufficient_function_decrease(double val)
{
    options_->line_search_sufficient_function_decrease_ = val;
    return *this;
}
solver_options_ceres_builder&
solver_options_ceres_builder::with_line_search_sufficient_curvature_decrease(double val)
{
    options_->line_search_sufficient_curvature_decrease_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_line_search_step_contraction(
    double val)
{
    options_->max_line_search_step_contraction_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_min_line_search_step_contraction(
    double val)
{
    options_->min_line_search_step_contraction_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_line_search_step_expansion(
    double val)
{
    options_->max_line_search_step_expansion_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_gradient_check_relative_precision(
    double val)
{
    options_->gradient_check_relative_precision_ = val;
    return *this;
}
solver_options_ceres_builder&
solver_options_ceres_builder::with_gradient_check_numeric_derivative_relative_step_size(double val)
{
    options_->gradient_check_numeric_derivative_relative_step_size_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_min_linear_solver_iterations(
    int val)
{
    options_->min_linear_solver_iterations_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_max_linear_solver_iterations(
    int val)
{
    options_->max_linear_solver_iterations_ = val;
    return *this;
}
solver_options_ceres_builder&
solver_options_ceres_builder::with_trust_region_minimizer_iterations_to_dump(
    const std::vector<int>& val)
{
    options_->trust_region_minimizer_iterations_to_dump_ = val;
    return *this;
}
solver_options_ceres_builder& solver_options_ceres_builder::with_trust_region_problem_dump_directory(
    const std::string& val)
{
    options_->trust_region_problem_dump_directory_ = val;
    return *this;
}
solver_options_ceres_builder&
solver_options_ceres_builder::with_trust_region_problem_dump_format_type(dump_format_enum val)
{
    options_->trust_region_problem_dump_format_type_ = val;
    return *this;
}

std::shared_ptr<const solver_options_ceres> solver_options_ceres_builder::build() const
{
    return options_;
}
}  // namespace solverslib
