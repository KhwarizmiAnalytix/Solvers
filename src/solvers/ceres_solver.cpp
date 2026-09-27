#include "solvers/ceres_solver.h"

#include "detail/support.h"

#if SOLVERS_HAS_CERES

#include <ceres/ceres.h>

#include "solvers/api/detail/evaluator.h"
#include "solver_options/solver_options_ceres.h"

#define DEBUG_AAD 0

namespace solverslib
{
class LambdaCostFunctor : public ceres::CostFunction
{
public:
    LambdaCostFunctor(
        ceres_solver::CostFunctionLambda     cost_function,
        ceres_solver::CostFunctionLambda_aad cost_function_aad,
        size_t                               num_parameters,
        size_t                               num_residuals)
        : cost_function_(std::move(cost_function)),
          cost_function_aad_(std::move(cost_function_aad)),
          num_parameters_(num_parameters),
          num_residuals_(num_residuals)
    {
        // Set the number of residuals and the size of each parameter block
        set_num_residuals((int)num_residuals_);
        mutable_parameter_block_sizes()->push_back((int)num_parameters_);
    }

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override
    {
        vector_type params_double = to_vector_type(parameters[0], num_parameters_);

        vector_type residual_values = make_vector(num_residuals_);
        cost_function_(params_double, residual_values);
        copy_into(residuals, num_residuals_, residual_values);

#if DEBUG_AAD
        double bump = 0.00001;

        auto cost_function_bump = [this, bump](vector_type const& x, matrix_type& dy_dx)
        {
            auto number_of_parameters = x.size();

            SOLVERS_CHECK(dy_dx.cols() == number_of_parameters);

            auto number_of_targets = dy_dx.rows();

            vector_type y_plus  = make_vector(number_of_targets);
            vector_type y_minus = make_vector(number_of_targets);

            vector_type x_tmp = make_vector(number_of_parameters);
            x_tmp             = x;

            for (size_t i = 0; i < number_of_parameters; ++i)
            {
                x_tmp[i] += bump;

                cost_function_(x_tmp, y_plus);

                x_tmp[i] -= 2 * bump;
                cost_function_(x_tmp, y_minus);

                for (size_t j = 0; j < y_plus.size(); ++j)
                {
                    dy_dx(j, i) = 0.5 * (y_plus[j] - y_minus[j]) / bump;
                }

                x_tmp[i] = x[i];
            }
        };
#endif

        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            matrix_type grad = make_matrix(num_residuals_, num_parameters_);
            cost_function_aad_(params_double, grad);
            copy_row_major(jacobians[0], grad);

#if DEBUG_AAD
            matrix_type grad2 = make_matrix(num_residuals_, num_parameters_);
            cost_function_bump(params_double, grad2);
#endif

            return true;
        }

        return true;
    }

private:
    ceres_solver::CostFunctionLambda     cost_function_;
    ceres_solver::CostFunctionLambda_aad cost_function_aad_;
    size_t                               num_residuals_;
    size_t                               num_parameters_;
};

// Cost function adapter for evaluator-based providers (e.g., Ceres AD)
class CeresProviderCostFunction : public ceres::CostFunction
{
public:
    explicit CeresProviderCostFunction(
        std::unique_ptr<api::detail::residual_evaluator> evaluator,
        size_t                                            num_residuals)
        : evaluator_(std::move(evaluator))
    {
        set_num_residuals(static_cast<int>(num_residuals));
        mutable_parameter_block_sizes()->push_back(
            static_cast<int>(evaluator_->metadata().num_parameters));
    }

    bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const override
    {
        try
        {
            const auto& meta = evaluator_->metadata();
            vector_type x = to_vector_type(parameters[0], meta.num_parameters);
            vector_type r = make_vector(meta.num_residuals);
            matrix_type* jac = nullptr;

            // Prepare Jacobian buffer if requested
            if (jacobians && jacobians[0])
            {
                jac = new matrix_type(make_matrix(meta.num_residuals, meta.num_parameters));
            }

            // Evaluate through provider
            auto status = evaluator_->evaluate(x, r, jac);
            if (status != api::detail::evaluation_status::ok)
            {
                delete jac;
                return false;  // Signal Ceres to reject this trial point
            }

            // Copy residuals back
            for (std::size_t i = 0; i < r.size(); ++i)
            {
                residuals[i] = r[i];
            }

            // Copy Jacobian if requested
            if (jac)
            {
                copy_row_major(jacobians[0], *jac);
                delete jac;
            }

            return true;
        }
        catch (const std::exception& e)
        {
            SOLVERS_LOG_ERROR("Evaluator exception: {}", e.what());
            return false;
        }
        catch (...)
        {
            SOLVERS_LOG_ERROR("Evaluator unknown exception");
            return false;
        }
    }

private:
    mutable std::unique_ptr<api::detail::residual_evaluator> evaluator_;
};

void update_options(
    ceres::Solver::Options& output_options, const solver_options_ceres& input_options)
{
    // General settings
    output_options.minimizer_type =
        static_cast<ceres::MinimizerType>(input_options.minimizer_type());

    output_options.line_search_direction_type =
        static_cast<ceres::LineSearchDirectionType>(input_options.line_search_direction_type());

    output_options.line_search_interpolation_type = static_cast<ceres::LineSearchInterpolationType>(
        input_options.line_search_interpolation_type());

    output_options.line_search_type =
        static_cast<ceres::LineSearchType>(input_options.line_search_type());

    output_options.trust_region_strategy_type =
        static_cast<ceres::TrustRegionStrategyType>(input_options.trust_region_strategy_type());

    output_options.dogleg_type = static_cast<ceres::DoglegType>(input_options.dogleg_type());

    // Iteration settings
    output_options.max_num_iterations           = input_options.max_num_iterations();
    output_options.function_tolerance           = input_options.function_tolerance();
    output_options.gradient_tolerance           = input_options.gradient_tolerance();
    output_options.parameter_tolerance          = input_options.parameter_tolerance();
    output_options.minimizer_progress_to_stdout = false;
    output_options.update_state_every_iteration = false;

    // Linear solver settings
    output_options.linear_solver_type =
        static_cast<ceres::LinearSolverType>(input_options.linear_solver_type());

    output_options.linear_solver_ordering_type =
        static_cast<ceres::LinearSolverOrderingType>(input_options.linear_solver_ordering_type());

    output_options.dense_linear_algebra_library_type =
        static_cast<ceres::DenseLinearAlgebraLibraryType>(
            input_options.dense_linear_algebra_library_type());

    output_options.sparse_linear_algebra_library_type =
        static_cast<ceres::SparseLinearAlgebraLibraryType>(
            input_options.sparse_linear_algebra_library_type());

    output_options.preconditioner_type =
        static_cast<ceres::PreconditionerType>(input_options.preconditioner_type());

    output_options.visibility_clustering_type =
        static_cast<ceres::VisibilityClusteringType>(input_options.visibility_clustering_type());

    // Trust region settings
    output_options.initial_trust_region_radius = input_options.initial_trust_region_radius();
    output_options.max_trust_region_radius     = input_options.max_trust_region_radius();
    output_options.min_trust_region_radius     = input_options.min_trust_region_radius();
    output_options.min_relative_decrease       = input_options.min_relative_decrease();
    output_options.eta                         = input_options.eta();

    // Inner iteration settings
    output_options.use_inner_iterations      = input_options.use_inner_iterations();
    output_options.inner_iteration_tolerance = input_options.inner_iteration_tolerance();

    // Jacobian scaling and sparsity
    output_options.jacobi_scaling   = input_options.jacobi_scaling();
    output_options.dynamic_sparsity = input_options.dynamic_sparsity();

    // Mixed precision settings
    output_options.use_mixed_precision_solves    = input_options.use_mixed_precision_solves();
    output_options.max_num_refinement_iterations = input_options.max_num_refinement_iterations();

    // Line search settings
    output_options.min_line_search_step_size = input_options.min_line_search_step_size();
    output_options.line_search_sufficient_function_decrease =
        input_options.line_search_sufficient_function_decrease();
    output_options.line_search_sufficient_curvature_decrease =
        input_options.line_search_sufficient_curvature_decrease();
    output_options.max_line_search_step_contraction =
        input_options.max_line_search_step_contraction();
    output_options.min_line_search_step_contraction =
        input_options.min_line_search_step_contraction();
    output_options.max_line_search_step_expansion = input_options.max_line_search_step_expansion();

    // Debugging and logging
    output_options.check_gradients = false;
    output_options.gradient_check_relative_precision =
        input_options.gradient_check_relative_precision();
    output_options.gradient_check_numeric_derivative_relative_step_size =
        input_options.gradient_check_numeric_derivative_relative_step_size();

    // Time and iteration limits
    output_options.max_solver_time_in_seconds   = input_options.max_solver_time_in_seconds();
    output_options.num_threads                  = input_options.num_threads();
    output_options.min_linear_solver_iterations = input_options.min_linear_solver_iterations();
    output_options.max_linear_solver_iterations = input_options.max_linear_solver_iterations();

    // Dumping output_options
    output_options.trust_region_minimizer_iterations_to_dump =
        input_options.trust_region_minimizer_iterations_to_dump();
    output_options.trust_region_problem_dump_directory =
        input_options.trust_region_problem_dump_directory();
    output_options.trust_region_problem_dump_format_type =
        static_cast<ceres::DumpFormatType>(input_options.trust_region_problem_dump_format_type());
}

//void update_options(ceres::Solver::Options& output_options, const solver_options_ceres& input_options)
//{
//    // General settings
//    output_options.minimizer_type = static_cast<ceres::MinimizerType>(input_options.minimizer_type());
//
//    output_options.line_search_direction_type =
//        static_cast<ceres::LineSearchDirectionType>(input_options.line_search_direction_type_);
//
//    output_options.line_search_interpolation_type =
//        static_cast<ceres::LineSearchInterpolationType>(input_options.line_search_interpolation_type_);
//
//    output_options.line_search_type = static_cast<ceres::LineSearchType>(input_options.line_search_type_);
//
//    output_options.trust_region_strategy_type =
//        static_cast<ceres::TrustRegionStrategyType>(input_options.trust_region_strategy_type_);
//
//    output_options.dogleg_type = static_cast<ceres::DoglegType>(input_options.dogleg_type_);
//
//    // Iteration settings
//    output_options.max_num_iterations           = input_options.max_num_iterations();
//    output_options.function_tolerance           = input_options.function_tolerance();
//    output_options.gradient_tolerance           = input_options.gradient_tolerance();
//    output_options.parameter_tolerance          = input_options.parameter_tolerance();
//    output_options.minimizer_progress_to_stdout = false;
//    output_options.update_state_every_iteration = false;
//
//    // Linear solver settings
//    output_options.linear_solver_type = static_cast<ceres::LinearSolverType>(input_options.linear_solver_type_);
//
//    output_options.linear_solver_ordering_type =
//        static_cast<ceres::LinearSolverOrderingType>(input_options.linear_solver_ordering_type_);
//
//    output_options.dense_linear_algebra_library_type = static_cast<ceres::DenseLinearAlgebraLibraryType>(
//        input_options.dense_linear_algebra_library_type_);
//
//    output_options.sparse_linear_algebra_library_type = static_cast<ceres::SparseLinearAlgebraLibraryType>(
//        input_options.sparse_linear_algebra_library_type_);
//
//    output_options.preconditioner_type =
//        static_cast<ceres::PreconditionerType>(input_options.preconditioner_type_);
//
//    output_options.visibility_clustering_type =
//        static_cast<ceres::VisibilityClusteringType>(input_options.visibility_clustering_type_);
//
//    // Trust region settings
//    output_options.initial_trust_region_radius = input_options.initial_trust_region_radius_;
//    output_options.max_trust_region_radius     = input_options.max_trust_region_radius_;
//    output_options.min_trust_region_radius     = input_options.min_trust_region_radius_;
//    output_options.min_relative_decrease       = input_options.min_relative_decrease_;
//    output_options.eta                         = input_options.eta_;
//
//    // Inner iteration settings
//    output_options.use_inner_iterations      = input_options.use_inner_iterations_;
//    output_options.inner_iteration_tolerance = input_options.inner_iteration_tolerance_;
//
//    // Jacobian scaling and sparsity
//    output_options.jacobi_scaling   = input_options.jacobi_scaling_;
//    output_options.dynamic_sparsity = input_options.dynamic_sparsity_;
//
//    // Mixed precision settings
//    output_options.use_mixed_precision_solves    = input_options.use_mixed_precision_solves_;
//    output_options.max_num_refinement_iterations = input_options.max_num_refinement_iterations_;
//
//    // Line search settings
//    output_options.min_line_search_step_size = input_options.min_line_search_step_size_;
//    output_options.line_search_sufficient_function_decrease =
//        input_options.line_search_sufficient_function_decrease_;
//    output_options.line_search_sufficient_curvature_decrease =
//        input_options.line_search_sufficient_curvature_decrease_;
//    output_options.max_line_search_step_contraction = input_options.max_line_search_step_contraction_;
//    output_options.min_line_search_step_contraction = input_options.min_line_search_step_contraction_;
//    output_options.max_line_search_step_expansion   = input_options.max_line_search_step_expansion_;
//
//    // Debugging and logging
//    output_options.check_gradients                   = false;
//    output_options.gradient_check_relative_precision = input_options.gradient_check_relative_precision_;
//    output_options.gradient_check_numeric_derivative_relative_step_size =
//        input_options.gradient_check_numeric_derivative_relative_step_size_;
//
//    // Time and iteration limits
//    output_options.max_solver_time_in_seconds   = input_options.max_solver_time_in_seconds_;
//    output_options.num_threads                  = input_options.num_threads_;
//    output_options.min_linear_solver_iterations = input_options.min_linear_solver_iterations_;
//    output_options.max_linear_solver_iterations = input_options.max_linear_solver_iterations_;
//
//    // Dumping output_options
//    output_options.trust_region_minimizer_iterations_to_dump =
//        input_options.trust_region_minimizer_iterations_to_dump_;
//    output_options.trust_region_problem_dump_directory = input_options.trust_region_problem_dump_directory_;
//    output_options.trust_region_problem_dump_format_type =
//        static_cast<ceres::DumpFormatType>(input_options.trust_region_problem_dump_format_type_);
//}
}  // namespace solverslib
#endif

namespace solverslib
{
ceres_solver::ceres_solver(
    size_t                     num_parameters,
    size_t                     num_residuals,
    CostFunctionLambda         cost_function,
    CostFunctionLambda_aad     cost_function_aad,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds)
    : cost_function_(std::move(cost_function)),
      cost_function_aad_(std::move(cost_function_aad)),
      provider_(nullptr),
      lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      num_parameters_(num_parameters),
      num_residuals_(num_residuals)
{
}

// New constructor with provider support
ceres_solver::ceres_solver(
    size_t                                                    num_parameters,
    size_t                                                    num_residuals,
    CostFunctionLambda                                        cost_function,
    std::shared_ptr<const api::detail::provider_factory>     provider,
    const std::vector<double>&                                lower_bounds,
    const std::vector<double>&                                upper_bounds)
    : cost_function_(std::move(cost_function)),
      cost_function_aad_(nullptr),
      provider_(std::move(provider)),
      lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      num_parameters_(num_parameters),
      num_residuals_(num_residuals)
{
}

bool ceres_solver::is_supported()
{
#if SOLVERS_HAS_CERES
    return true;
#else
    return false;
#endif
};

bool ceres_solver::solve(
    SOLVERS_UNUSED std::vector<double>&        parameters,
    SOLVERS_UNUSED const solver_options_ceres& options)
{
#if SOLVERS_HAS_CERES
    SOLVERS_CHECK(parameters.size() == num_parameters_);

    ceres::Problem problem;
    ceres::CostFunction* cost_function = nullptr;

    // ============================================================================
    // Phase 1: Derivative selection - use provider if available, else fallback
    // ============================================================================
    std::unique_ptr<api::detail::residual_evaluator> evaluator;

    if (provider_)
    {
        // Use evaluator-based provider (e.g., Ceres AD)
        // Validate provider dimensions match problem dimensions
        const auto& meta = provider_->metadata();
        SOLVERS_CHECK(
            meta.num_parameters == num_parameters_ && meta.num_residuals == num_residuals_,
            "Provider dimensions do not match problem dimensions");

        evaluator = provider_->create_evaluator();
        cost_function =
            new CeresProviderCostFunction(std::move(evaluator), num_residuals_);
    }
    else
    {
        // Fallback to legacy callback path
        // Set up finite differences if no Jacobian callback provided
        if (cost_function_aad_ == nullptr)
        {
            double bump = 1e-8;

            cost_function_aad_ = [this, bump](vector_type const& x, matrix_type& dy_dx)
            {
                auto number_of_parameters = x.size();

                SOLVERS_CHECK(dy_dx.cols() == number_of_parameters);

                auto number_of_targets = dy_dx.rows();

                vector_type y_plus  = make_vector(number_of_targets);
                vector_type y_minus = make_vector(number_of_targets);

                vector_type x_tmp = make_vector(number_of_parameters);
                x_tmp             = x;

                for (size_t i = 0; i < number_of_parameters; ++i)
                {
                    x_tmp[i] += bump;

                    cost_function_(x_tmp, y_plus);

                    x_tmp[i] -= 2 * bump;
                    cost_function_(x_tmp, y_minus);

                    for (size_t j = 0; j < y_plus.size(); ++j)
                    {
                        dy_dx(j, i) = 0.5 * (y_plus[j] - y_minus[j]) / bump;
                    }

                    x_tmp[i] = x[i];
                }
            };
        }

        cost_function = new LambdaCostFunctor(
            cost_function_,
            cost_function_aad_,
            num_parameters_,
            num_residuals_);
    }

    problem.AddResidualBlock(cost_function, nullptr, parameters.data());

    // ============================================================================
    // Phase 2: Apply bounds independently (lower and upper are optional)
    // ============================================================================
    if (lower_bounds_.size() > 0)
    {
        SOLVERS_CHECK(
            lower_bounds_.size() >= num_parameters_,
            "Lower bounds size must be at least num_parameters");

        for (std::size_t i = 0; i < num_parameters_; ++i)
        {
            problem.SetParameterLowerBound(parameters.data(), static_cast<int>(i), lower_bounds_[i]);
        }
    }

    if (upper_bounds_.size() > 0)
    {
        SOLVERS_CHECK(
            upper_bounds_.size() >= num_parameters_,
            "Upper bounds size must be at least num_parameters");

        for (std::size_t i = 0; i < num_parameters_; ++i)
        {
            problem.SetParameterUpperBound(parameters.data(), static_cast<int>(i), upper_bounds_[i]);
        }
    }

    ceres::Solver::Options output_options;
    update_options(output_options, options);
#if 0
    // General settings
    output_options.minimizer_type                 = ceres::MinimizerType::TRUST_REGION;
    output_options.line_search_direction_type     = ceres::LineSearchDirectionType::STEEPEST_DESCENT;
    output_options.line_search_interpolation_type = ceres::LineSearchInterpolationType::BISECTION;
    output_options.line_search_type               = ceres::LineSearchType::WOLFE;
    output_options.trust_region_strategy_type     = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
    output_options.dogleg_type                    = ceres::DoglegType::SUBSPACE_DOGLEG;

    // Iteration settings
    output_options.max_num_iterations           = 100;
    output_options.function_tolerance           = 1e-14;
    output_options.gradient_tolerance           = 1e-14;
    output_options.parameter_tolerance          = 1e-14;
    output_options.minimizer_progress_to_stdout = false;
    output_options.update_state_every_iteration = false;

    // Linear solver settings
    output_options.linear_solver_type                = ceres::LinearSolverType::DENSE_SCHUR;
    output_options.linear_solver_ordering_type       = ceres::LinearSolverOrderingType::NESDIS;
    output_options.dense_linear_algebra_library_type = ceres::DenseLinearAlgebraLibraryType::EIGEN;
    output_options.sparse_linear_algebra_library_type =
        ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;
    output_options.preconditioner_type        = ceres::PreconditionerType::JACOBI;
    output_options.visibility_clustering_type = ceres::VisibilityClusteringType::CANONICAL_VIEWS;

    // Trust region settings
    output_options.initial_trust_region_radius = 0.1;
    output_options.max_trust_region_radius     = 1e16;
    output_options.min_trust_region_radius     = 1e-32;
    output_options.min_relative_decrease       = 1e-3;
    output_options.eta                         = 0.0001;

    // Inner iteration settings
    output_options.use_inner_iterations      = false;
    output_options.inner_iteration_tolerance = 1e-14;

    // Jacobian scaling and sparsity
    output_options.jacobi_scaling   = true;
    output_options.dynamic_sparsity = false;

    // Mixed precision settings
    output_options.use_mixed_precision_solves    = false;
    output_options.max_num_refinement_iterations = 0;

    // Line search settings
    output_options.min_line_search_step_size                 = 1e-9;
    output_options.line_search_sufficient_function_decrease  = 1e-4;
    output_options.line_search_sufficient_curvature_decrease = 0.9;
    output_options.max_line_search_step_contraction          = 1e-3;
    output_options.min_line_search_step_contraction          = 0.6;
    output_options.max_line_search_step_expansion            = 10.0;

    // Debugging and logging
    output_options.check_gradients                                      = false;
    output_options.gradient_check_relative_precision                    = 1e-8;
    output_options.gradient_check_numeric_derivative_relative_step_size = 1e-6;

    // Time and iteration limits
    output_options.max_solver_time_in_seconds   = 1e9;
    output_options.num_threads                  = 1;
    output_options.min_linear_solver_iterations = 0;
    output_options.max_linear_solver_iterations = 500;

    // Dumping output_options
    output_options.trust_region_minimizer_iterations_to_dump = {};
    output_options.trust_region_problem_dump_directory       = "/tmp";
    output_options.trust_region_problem_dump_format_type     = ceres::DumpFormatType::TEXTFILE;

    // Callbacks
    output_options.callbacks = {};
#endif

    ceres::Solver::Summary summary;
    ceres::Solve(output_options, &problem, &summary);

    if (options.verbose())
    {
        SOLVERS_LOG_INFO("ceres solver summary: {}", summary.BriefReport());
    }

    return summary.IsSolutionUsable();
#else
    SOLVERS_NOT_IMPLEMENTED("ceres solver not supported");
#endif
}
}  // namespace solverslib
