#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "detail/support.h"
#include "solver_options/solver_options.h"

namespace solverslib
{
enum class linear_solver_enum : int16_t
{
    DENSE_NORMAL_CHOLESKY,
    DENSE_QR,
    SPARSE_NORMAL_CHOLESKY,
    DENSE_SCHUR,
    SPARSE_SCHUR,
    ITERATIVE_SCHUR,
    CGNR
};

enum class preconditioner_enum : int16_t
{
    IDENTITY,
    JACOBI,
    SCHUR_JACOBI,
    SCHUR_POWER_SERIES_EXPANSION,
    CLUSTER_JACOBI,
    CLUSTER_TRIDIAGONAL,
    SUBSET
};

enum class visibility_clustering_enum : int16_t
{
    CANONICAL_VIEWS,
    SINGLE_LINKAGE
};

enum class sparse_linear_algebra_library_enum : int16_t
{
    SUITE_SPARSE     = 0,
    EIGEN_SPARSE     = 1,
    ACCELERATE_SPARSE = 2,
    CUDA_SPARSE      = 3,
    NO_SPARSE        = 4,
    INVALID          = -1
};

enum class linear_solver_ordering_enum : int16_t
{
    AMD    = 0,
    NESDIS = 1,
    INVALID = -1
};

enum class dense_linear_algebra_library_enum : int16_t
{
    EIGEN  = 0,
    LAPACK = 1,
    CUDA   = 2,
    INVALID = -1
};

enum class logging_enum : int16_t
{
    SILENT                  = 0,
    PER_MINIMIZER_ITERATION = 1,
    INVALID                 = -1
};

enum class minimizer_enum : int16_t
{
    LINE_SEARCH  = 0,
    TRUST_REGION = 1,
    INVALID      = -1
};

enum class line_search_direction_enum : int16_t
{
    STEEPEST_DESCENT            = 0,
    NONLINEAR_CONJUGATE_GRADIENT = 1,
    lbfgs_solver                = 2,
    BFGS                        = 3,
};

enum class nonlinear_conjugate_gradient_enum : int16_t
{
    FLETCHER_REEVES  = 0,
    POLAK_RIBIERE    = 1,
    HESTENES_STIEFEL = 2,
};

enum class line_search_enum : int16_t
{
    ARMIJO = 0,
    WOLFE  = 1,
    INVALID = -1
};

enum class trust_region_strategy_enum : int16_t
{
    LEVENBERG_MARQUARDT = 0,
    DOGLEG              = 1,
    INVALID             = -1
};

enum class dogleg_enum : int16_t
{
    TRADITIONAL_DOGLEG = 0,
    SUBSPACE_DOGLEG    = 1,
    INVALID            = -1
};

enum class callback_return_enum : int16_t
{
    SOLVER_CONTINUE              = 0,
    SOLVER_ABORT                 = 1,
    SOLVER_TERMINATE_SUCCESSFULLY = 2,
    INVALID                      = -1
};

enum class dimension_enum : int16_t
{
    DYNAMIC = 0,
    INVALID = -1
};

enum class numeric_diff_method_enum : int16_t
{
    CENTRAL = 0,
    FORWARD = 1,
    RIDDERS = 2,
    INVALID = -1
};

enum class line_search_interpolation_enum : int16_t
{
    BISECTION = 0,
    QUADRATIC = 1,
    CUBIC     = 2,
    INVALID   = -1
};

enum class covariance_algorithm_enum : int16_t
{
    DENSE_SVD = 0,
    SPARSE_QR = 1,
    INVALID   = -1
};

enum class dump_format_enum : int16_t
{
    CONSOLE  = 0,
    TEXTFILE = 1,
    INVALID  = -1
};
}  // namespace solverslib

namespace solverslib
{
class SOLVER_VISIBILITY solver_options_ceres : public solver_options
{
    friend class solver_options_ceres_builder;

public:
    SOLVER_API visibility_clustering_enum         visibility_clustering_type() const;
    SOLVER_API preconditioner_enum                preconditioner_type() const;
    SOLVER_API sparse_linear_algebra_library_enum sparse_linear_algebra_library_type() const;
    SOLVER_API dense_linear_algebra_library_enum  dense_linear_algebra_library_type() const;
    SOLVER_API linear_solver_enum                 linear_solver_type() const;
    SOLVER_API linear_solver_ordering_enum        linear_solver_ordering_type() const;
    SOLVER_API dogleg_enum                        dogleg_type() const;
    SOLVER_API trust_region_strategy_enum         trust_region_strategy_type() const;
    SOLVER_API line_search_interpolation_enum     line_search_interpolation_type() const;
    SOLVER_API minimizer_enum                     minimizer_type() const;
    SOLVER_API line_search_direction_enum         line_search_direction_type() const;
    SOLVER_API line_search_enum                   line_search_type() const;
    SOLVER_API nonlinear_conjugate_gradient_enum  nonlinear_conjugate_gradient_type() const;
    SOLVER_API numeric_diff_method_enum           numeric_diff_method() const;
    SOLVER_API int                                max_lbfgs_rank() const;
    SOLVER_API double                             initial_trust_region_radius() const;
    SOLVER_API double                             max_trust_region_radius() const;
    SOLVER_API double                             min_trust_region_radius() const;
    SOLVER_API double                             min_relative_decrease() const;
    SOLVER_API double                             eta() const;
    SOLVER_API bool                               use_inner_iterations() const;
    SOLVER_API double                             inner_iteration_tolerance() const;
    SOLVER_API bool                               jacobi_scaling() const;
    SOLVER_API bool                               dynamic_sparsity() const;
    SOLVER_API bool                               use_mixed_precision_solves() const;
    SOLVER_API int                                max_num_refinement_iterations() const;
    SOLVER_API double                             min_line_search_step_size() const;
    SOLVER_API double line_search_sufficient_function_decrease() const;
    SOLVER_API double line_search_sufficient_curvature_decrease() const;
    SOLVER_API double max_line_search_step_contraction() const;
    SOLVER_API double min_line_search_step_contraction() const;
    SOLVER_API double max_line_search_step_expansion() const;
    SOLVER_API bool   check_gradients() const;
    SOLVER_API double gradient_check_relative_precision() const;
    SOLVER_API double gradient_check_numeric_derivative_relative_step_size() const;
    SOLVER_API double max_solver_time_in_seconds() const;
    SOLVER_API int    num_threads() const;
    SOLVER_API int    min_linear_solver_iterations() const;
    SOLVER_API int    max_linear_solver_iterations() const;
    SOLVER_API const std::vector<int>& trust_region_minimizer_iterations_to_dump() const;
    SOLVER_API const std::string&      trust_region_problem_dump_directory() const;
    SOLVER_API dump_format_enum        trust_region_problem_dump_format_type() const;

private:
    solver_options_ceres();

    visibility_clustering_enum visibility_clustering_type_ =
        visibility_clustering_enum::CANONICAL_VIEWS;
    preconditioner_enum                preconditioner_type_ = preconditioner_enum::JACOBI;
    sparse_linear_algebra_library_enum sparse_linear_algebra_library_type_ =
        sparse_linear_algebra_library_enum::EIGEN_SPARSE;
    dense_linear_algebra_library_enum dense_linear_algebra_library_type_ =
        dense_linear_algebra_library_enum::EIGEN;
    linear_solver_enum          linear_solver_type_ = linear_solver_enum::SPARSE_NORMAL_CHOLESKY;
    linear_solver_ordering_enum linear_solver_ordering_type_ = linear_solver_ordering_enum::AMD;
    dogleg_enum                 dogleg_type_                 = dogleg_enum::TRADITIONAL_DOGLEG;
    trust_region_strategy_enum  trust_region_strategy_type_ =
        trust_region_strategy_enum::LEVENBERG_MARQUARDT;
    line_search_interpolation_enum line_search_interpolation_type_ =
        line_search_interpolation_enum::CUBIC;
    minimizer_enum             minimizer_type_ = minimizer_enum::TRUST_REGION;
    line_search_direction_enum line_search_direction_type_ =
        line_search_direction_enum::lbfgs_solver;
    line_search_enum                  line_search_type_ = line_search_enum::WOLFE;
    nonlinear_conjugate_gradient_enum nonlinear_conjugate_gradient_type_ =
        nonlinear_conjugate_gradient_enum::FLETCHER_REEVES;
    numeric_diff_method_enum numeric_diff_method_           = numeric_diff_method_enum::CENTRAL;
    int                      max_lbfgs_rank_                = 20;
    double                   initial_trust_region_radius_   = 0.1;
    double                   max_trust_region_radius_       = 1e16;
    double                   min_trust_region_radius_       = 1e-32;
    double                   min_relative_decrease_         = 1e-3;
    double                   eta_                           = 0.0001;
    bool                     use_inner_iterations_          = false;
    double                   inner_iteration_tolerance_     = 1e-14;
    bool                     jacobi_scaling_                = true;
    bool                     dynamic_sparsity_              = false;
    bool                     use_mixed_precision_solves_    = false;
    int                      max_num_refinement_iterations_ = 0;
    double                   min_line_search_step_size_     = 1e-9;
    double                   line_search_sufficient_function_decrease_             = 1e-4;
    double                   line_search_sufficient_curvature_decrease_            = 0.9;
    double                   max_line_search_step_contraction_                     = 1e-3;
    double                   min_line_search_step_contraction_                     = 0.6;
    double                   max_line_search_step_expansion_                       = 10.0;
    bool                     check_gradients_                                      = false;
    double                   gradient_check_relative_precision_                    = 1e-8;
    double                   gradient_check_numeric_derivative_relative_step_size_ = 1e-6;
    double                   max_solver_time_in_seconds_                           = 1e9;
    int                      num_threads_                                          = 1;
    int                      min_linear_solver_iterations_                         = 0;
    int                      max_linear_solver_iterations_                         = 500;
    std::vector<int>         trust_region_minimizer_iterations_to_dump_            = {};
    std::string              trust_region_problem_dump_directory_                  = "/tmp";
    dump_format_enum         trust_region_problem_dump_format_type_ = dump_format_enum::TEXTFILE;
};

class SOLVER_VISIBILITY solver_options_ceres_builder
{
public:
    SOLVER_API solver_options_ceres_builder();

    SOLVER_API solver_options_ceres_builder& with_max_iterations(int val);
    SOLVER_API solver_options_ceres_builder& with_function_tolerance(double val);
    SOLVER_API solver_options_ceres_builder& with_gradient_tolerance(double val);
    SOLVER_API solver_options_ceres_builder& with_parameter_tolerance(double val);
    SOLVER_API solver_options_ceres_builder& with_verbose(bool val = true);
    SOLVER_API solver_options_ceres_builder& with_log_file(const std::string& val);
    SOLVER_API solver_options_ceres_builder& with_linear_solver_type(linear_solver_enum val);
    SOLVER_API solver_options_ceres_builder& with_preconditioner_type(preconditioner_enum val);
    SOLVER_API solver_options_ceres_builder& with_minimizer_type(minimizer_enum val);
    SOLVER_API solver_options_ceres_builder& with_trust_region_strategy_type(
        trust_region_strategy_enum val);
    SOLVER_API solver_options_ceres_builder& with_num_threads(int val);
    SOLVER_API solver_options_ceres_builder& with_max_solver_time_in_seconds(double val);
    SOLVER_API solver_options_ceres_builder& with_dogleg_type(dogleg_enum val);
    SOLVER_API solver_options_ceres_builder& with_line_search_direction_type(
        line_search_direction_enum val);
    SOLVER_API solver_options_ceres_builder& with_line_search_type(line_search_enum val);
    SOLVER_API solver_options_ceres_builder& with_nonlinear_conjugate_gradient_type(
        nonlinear_conjugate_gradient_enum val);
    SOLVER_API solver_options_ceres_builder& with_line_search_interpolation_type(
        line_search_interpolation_enum val);
    SOLVER_API solver_options_ceres_builder& with_numeric_diff_method(
        numeric_diff_method_enum val);
    SOLVER_API solver_options_ceres_builder& with_sparse_linear_algebra_library_type(
        sparse_linear_algebra_library_enum val);
    SOLVER_API solver_options_ceres_builder& with_dense_linear_algebra_library_type(
        dense_linear_algebra_library_enum val);
    SOLVER_API solver_options_ceres_builder& with_linear_solver_ordering_type(
        linear_solver_ordering_enum val);
    SOLVER_API solver_options_ceres_builder& with_visibility_clustering_type(
        visibility_clustering_enum val);
    SOLVER_API solver_options_ceres_builder& with_initial_trust_region_radius(double val);
    SOLVER_API solver_options_ceres_builder& with_max_trust_region_radius(double val);
    SOLVER_API solver_options_ceres_builder& with_min_trust_region_radius(double val);
    SOLVER_API solver_options_ceres_builder& with_min_relative_decrease(double val);
    SOLVER_API solver_options_ceres_builder& with_eta(double val);
    SOLVER_API solver_options_ceres_builder& with_use_inner_iterations(bool val = true);
    SOLVER_API solver_options_ceres_builder& with_inner_iteration_tolerance(double val);
    SOLVER_API solver_options_ceres_builder& with_jacobi_scaling(bool val = true);
    SOLVER_API solver_options_ceres_builder& with_dynamic_sparsity(bool val = true);
    SOLVER_API solver_options_ceres_builder& with_use_mixed_precision_solves(bool val = true);
    SOLVER_API solver_options_ceres_builder& with_max_num_refinement_iterations(int val);
    SOLVER_API solver_options_ceres_builder& with_max_lbfgs_rank(int val);
    SOLVER_API solver_options_ceres_builder& with_min_line_search_step_size(double val);
    SOLVER_API solver_options_ceres_builder& with_line_search_sufficient_function_decrease(
        double val);
    SOLVER_API solver_options_ceres_builder& with_line_search_sufficient_curvature_decrease(
        double val);
    SOLVER_API solver_options_ceres_builder& with_max_line_search_step_contraction(double val);
    SOLVER_API solver_options_ceres_builder& with_min_line_search_step_contraction(double val);
    SOLVER_API solver_options_ceres_builder& with_max_line_search_step_expansion(double val);
    SOLVER_API solver_options_ceres_builder& with_check_gradients(bool val = true);
    SOLVER_API solver_options_ceres_builder& with_gradient_check_relative_precision(double val);
    SOLVER_API solver_options_ceres_builder&
    with_gradient_check_numeric_derivative_relative_step_size(double val);
    SOLVER_API solver_options_ceres_builder& with_min_linear_solver_iterations(int val);
    SOLVER_API solver_options_ceres_builder& with_max_linear_solver_iterations(int val);
    SOLVER_API solver_options_ceres_builder& with_trust_region_minimizer_iterations_to_dump(
        const std::vector<int>& val);
    SOLVER_API solver_options_ceres_builder& with_trust_region_problem_dump_directory(
        const std::string& val);
    SOLVER_API solver_options_ceres_builder& with_trust_region_problem_dump_format_type(
        dump_format_enum val);
    SOLVER_API solver_options_ceres_builder& with_aad_jacobian(bool val = true);

    SOLVER_API std::shared_ptr<const solver_options_ceres> build() const;

private:
    std::shared_ptr<solver_options_ceres> options_;
};

}  // namespace solverslib
