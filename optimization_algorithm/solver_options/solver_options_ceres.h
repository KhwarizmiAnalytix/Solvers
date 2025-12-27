#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "MathModule.h"
#include "common/pointer.h"
#include "common/serialization_macros.h"
#include "optimization_algorithm/solver_options/solver_options.h"

namespace quarisma
{
// TODO(keir): Considerably expand the explanations of each solver type.
enum class linear_solver_enum : int16_t
{
    // These solvers are for general rectangular systems formed from the
    // normal equations A'A x = A'b. They are direct solvers and do not
    // assume any special problem structure.

    // Solve the normal equations using a dense Cholesky solver; based
    // on Eigen.
    DENSE_NORMAL_CHOLESKY,

    // Solve the normal equations using a dense QR solver; based on
    // Eigen.
    DENSE_QR,

    // Solve the normal equations using a sparse cholesky solver;
    SPARSE_NORMAL_CHOLESKY,

    // Specialized solvers, specific to problems with a generalized
    // bi-partitite structure.

    // Solves the reduced linear system using a dense Cholesky solver;
    // based on Eigen.
    DENSE_SCHUR,

    // Solves the reduced linear system using a sparse Cholesky solver;
    // based on CHOLMOD.
    SPARSE_SCHUR,

    // Solves the reduced linear system using Conjugate Gradients, based
    // on a new Ceres implementation.  Suitable for large scale
    // problems.
    ITERATIVE_SCHUR,

    // Conjugate gradients on the normal equations.
    CGNR
};

enum class preconditioner_enum : int16_t
{
    // Trivial preconditioner - the identity matrix.
    IDENTITY,

    // Block diagonal of the Gauss-Newton Hessian.
    JACOBI,

    // Note: The following four preconditioners can only be used with
    // the ITERATIVE_SCHUR solver. They are well suited for Structure
    // from Motion problems.

    // Block diagonal of the Schur complement. This preconditioner may
    // only be used with the ITERATIVE_SCHUR solver.
    SCHUR_JACOBI,

    // Use power series expansion to approximate the inversion of Schur complement
    // as a preconditioner.
    SCHUR_POWER_SERIES_EXPANSION,

    // Visibility clustering based preconditioners.
    //
    // The following two preconditioners use the visibility structure of
    // the scene to determine the sparsity structure of the
    // preconditioner. This is done using a clustering algorithm. The
    // available visibility clustering algorithms are described below.
    CLUSTER_JACOBI,
    CLUSTER_TRIDIAGONAL,

    // Subset preconditioner is a general purpose preconditioner
    // linear least squares problems. Given a set of residual blocks,
    // it uses the corresponding subset of the rows of the Jacobian to
    // construct a preconditioner.
    //
    // Suppose the Jacobian J has been horizontally partitioned as
    //
    // J = [P]
    //     [Q]
    //
    // Where, Q is the set of rows corresponding to the residual
    // blocks in residual_blocks_for_subset_preconditioner.
    //
    // The preconditioner is the inverse of the matrix Q'Q.
    //
    // Obviously, the efficacy of the preconditioner depends on how
    // well the matrix Q approximates J'J, or how well the chosen
    // residual blocks approximate the non-linear least squares
    // problem.
    SUBSET
};

enum class visibility_clustering_enum : int16_t
{
    // Canonical views algorithm as described in
    //
    // "Scene Summarization for Online Image Collections", Ian Simon, Noah
    // Snavely, Steven M. Seitz, ICCV 2007.
    //
    // This clustering algorithm can be quite slow, but gives high
    // quality clusters. The original visibility based clustering paper
    // used this algorithm.
    CANONICAL_VIEWS,

    // The classic single linkage algorithm. It is extremely fast as
    // compared to CANONICAL_VIEWS, but can give slightly poorer
    // results. For problems with large number of cameras though, this
    // is generally a pretty good option.
    //
    // If you are using SCHUR_JACOBI preconditioner and have SuiteSparse
    // available, CLUSTER_JACOBI and CLUSTER_TRIDIAGONAL in combination
    // with the SINGLE_LINKAGE algorithm will generally give better
    // results.
    SINGLE_LINKAGE
};

enum class sparse_linear_algebra_library_enum : int16_t
{
    // High performance sparse Cholesky factorization and approximate
    // minimum degree ordering.
    SUITE_SPARSE = 0,

    // Eigen's sparse linear algebra routines. In particular Ceres uses
    // the Simplicial LDLT routines.
    EIGEN_SPARSE = 1,

    // Apple's Accelerate framework sparse linear algebra routines.
    ACCELERATE_SPARSE = 2,

    // Nvidia's cuDSS and cuSPARSE libraries.
    CUDA_SPARSE = 3,

    // No sparse linear solver should be used.  This does not necessarily
    // imply that Ceres was built without any sparse library, although that
    // is the likely use case, merely that one should not be used.
    NO_SPARSE = 4,

    INVALID = -1
};

// The order in which variables are eliminated in a linear solver
// can have a significant of impact on the efficiency and accuracy
// of the method. e.g., when doing sparse Cholesky factorization,
// there are matrices for which a good ordering will give a
// Cholesky factor with O(n) storage, where as a bad ordering will
// result in a completely dense factor.
//
// So sparse direct solvers like SPARSE_NORMAL_CHOLESKY and
// SPARSE_SCHUR and preconditioners like SUBSET, CLUSTER_JACOBI &
// CLUSTER_TRIDIAGONAL use a fill reducing ordering of the columns and
// rows of the matrix being factorized before actually the numeric
// factorization.
//
// This enum controls the class of algorithm used to compute this
// fill reducing ordering. There is no single algorithm that works
// on all matrices, so determining which algorithm works better is a
// matter of empirical experimentation.
enum class linear_solver_ordering_enum : int16_t
{
    // Approximate Minimum Degree.
    AMD = 0,
    // Nested Dissection.
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

// Logging options
// The options get progressively noisier.
enum class logging_enum : int16_t
{
    SILENT                  = 0,
    PER_MINIMIZER_ITERATION = 1,

    INVALID = -1
};

enum class minimizer_enum : int16_t
{
    LINE_SEARCH  = 0,
    TRUST_REGION = 1,

    INVALID = -1
};

enum class line_search_direction_enum : int16_t
{
    // Negative of the gradient.
    STEEPEST_DESCENT = 0,

    // A generalization of the Conjugate Gradient method to non-linear
    // functions. The generalization can be performed in a number of
    // different ways, resulting in a variety of search directions. The
    // precise choice of the non-linear conjugate gradient algorithm
    // used is determined by NonlinerConjuateGradientType.
    NONLINEAR_CONJUGATE_GRADIENT = 1,

    // BFGS, and it's limited memory approximation L-BFGS, are quasi-Newton
    // algorithms that approximate the Hessian matrix by iteratively refining
    // an initial estimate with rank-one updates using the gradient at each
    // iteration. They are a generalisation of the Secant method and satisfy
    // the Secant equation.  The Secant equation has an infinium of solutions
    // in multiple dimensions, as there are N*(N+1)/2 degrees of freedom in a
    // symmetric matrix but only N conditions are specified by the Secant
    // equation. The requirement that the Hessian approximation be positive
    // definite imposes another N additional constraints, but that still leaves
    // remaining degrees-of-freedom.  (L)BFGS methods uniquely determine the
    // approximate Hessian by imposing the additional constraints that the
    // approximation at the next iteration must be the 'closest' to the current
    // approximation (the nature of how this proximity is measured is actually
    // the defining difference between a family of quasi-Newton methods including
    // (L)BFGS & DFP). (L)BFGS is currently regarded as being the best known
    // general quasi-Newton method.
    //
    // The principal difference between BFGS and L-BFGS is that whilst BFGS
    // maintains a full, dense approximation to the (inverse) Hessian, L-BFGS
    // maintains only a window of the last M observations of the parameters and
    // gradients. Using this observation history, the calculation of the next
    // search direction can be computed without requiring the construction of the
    // full dense inverse Hessian approximation. This is particularly important
    // for problems with a large number of parameters, where storage of an N-by-N
    // matrix in memory would be prohibitive.
    //
    // For more details on BFGS see:
    //
    // Broyden, C.G., "The Convergence of a Class of Double-rank Minimization
    // Algorithms,"; J. Inst. Maths. Applics., Vol. 6, pp 76-90, 1970.
    //
    // Fletcher, R., "A New Approach to Variable Metric Algorithms,"
    // Computer Journal, Vol. 13, pp 317-322, 1970.
    //
    // Goldfarb, D., "A Family of Variable Metric Updates Derived by Variational
    // Means," Mathematics of Computing, Vol. 24, pp 23-26, 1970.
    //
    // Shanno, D.F., "Conditioning of Quasi-Newton Methods for Function
    // Minimization," Mathematics of Computing, Vol. 24, pp 647-656, 1970.
    //
    // For more details on L-BFGS see:
    //
    // Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited
    // Storage". Mathematics of Computation 35 (151): 773-782.
    //
    // Byrd, R. H.; Nocedal, J.; Schnabel, R. B. (1994).
    // "Representations of Quasi-Newton Matrices and their use in
    // Limited Memory Methods". Mathematical Programming 63 (4):
    // 129-156.
    //
    // A general reference for both methods:
    //
    // Nocedal J., Wright S., Numerical Optimization, 2nd Ed. Springer, 1999.
    lbfgs_solver = 2,
    BFGS         = 3,
};

// Nonlinear conjugate gradient methods are a generalization of the
// method of Conjugate Gradients for linear systems. The
// generalization can be carried out in a number of different ways
// leading to number of different rules for computing the search
// direction. Ceres provides a number of different variants. For more
// details see Numerical Optimization by Nocedal & Wright.
enum class nonlinear_conjugate_gradient_enum : int16_t
{
    FLETCHER_REEVES  = 0,
    POLAK_RIBIERE    = 1,
    HESTENES_STIEFEL = 2,
};

enum class line_search_enum : int16_t
{
    // Backtracking line search with polynomial interpolation or
    // bisection.
    ARMIJO = 0,
    WOLFE  = 1,

    INVALID = -1
};

// Ceres supports different strategies for computing the trust region
// step.
enum class trust_region_strategy_enum : int16_t
{
    // The default trust region strategy is to use the step computation
    // used in the Levenberg-Marquardt algorithm. For more details see
    // levenberg_marquardt_solver_strategy.h
    LEVENBERG_MARQUARDT = 0,

    // Powell's dogleg algorithm interpolates between the Cauchy point
    // and the Gauss-Newton step. It is particularly useful if the
    // LEVENBERG_MARQUARDT algorithm is making a large number of
    // unsuccessful steps. For more details see dogleg_strategy.h.
    //
    // NOTES:
    //
    // 1. This strategy has not been experimented with or tested as
    // extensively as LEVENBERG_MARQUARDT, and therefore it should be
    // considered EXPERIMENTAL for now.
    //
    // 2. For now this strategy should only be used with exact
    // factorization based linear solvers, i.e., SPARSE_SCHUR,
    // DENSE_SCHUR, DENSE_QR and SPARSE_NORMAL_CHOLESKY.
    DOGLEG = 1,

    INVALID = -1
};

// Ceres supports two different dogleg strategies.
// The "traditional" dogleg method by Powell and the
// "subspace" method described in
// R. H. Byrd, R. B. Schnabel, and G. A. Shultz,
// "Approximate solution of the trust region problem by minimization
//  over two-dimensional subspaces", Mathematical Programming,
// 40 (1988), pp. 247--263
enum class dogleg_enum : int16_t
{
    // The traditional approach constructs a dogleg path
    // consisting of two line segments and finds the furthest
    // point on that path that is still inside the trust region.
    TRADITIONAL_DOGLEG = 0,

    // The subspace approach finds the exact minimum of the model
    // constrained to the subspace spanned by the dogleg path.
    SUBSPACE_DOGLEG = 1,

    INVALID = -1
};

// Enums used by the IterationCallback instances to indicate to the
// solver whether it should continue solving, the user detected an
// error or the solution is good enough and the solver should
// terminate.
enum class callback_return_enum : int16_t
{
    // Continue solving to next iteration.
    SOLVER_CONTINUE = 0,

    // Terminate solver, and do not update the parameter blocks upon
    // return. Unless the user has set
    // Solver:Options:::update_state_every_iteration, in which case the
    // state would have been updated every iteration
    // anyways. Solver::Summary::termination_type is set to USER_ABORT.
    SOLVER_ABORT = 1,

    // Terminate solver, update state and
    // return. Solver::Summary::termination_type is set to USER_SUCCESS.
    SOLVER_TERMINATE_SUCCESSFULLY = 2,

    INVALID = -1
};

// For SizedCostFunction and AutoDiffCostFunction, DYNAMIC can be
// specified for the number of residuals. If specified, then the
// number of residuas for that cost function can vary at runtime.
enum class dimension_enum : int16_t
{
    DYNAMIC = 0,

    INVALID = -1
};

// The differentiation method used to compute numerical derivatives in
// NumericDiffCostFunction and DynamicNumericDiffCostFunction.
enum class numeric_diff_method_enum : int16_t
{
    // Compute central finite difference: f'(x) ~ (f(x+h) - f(x-h)) / 2h.
    CENTRAL = 0,

    // Compute forward finite difference: f'(x) ~ (f(x+h) - f(x)) / h.
    FORWARD = 1,

    // Adaptive numerical differentiation using Ridders' method. Provides more
    // accurate and robust derivatives at the expense of additional cost
    // function evaluations.
    RIDDERS = 2,

    INVALID = -1
};

enum class line_search_interpolation_enum : int16_t
{
    BISECTION = 0,
    QUADRATIC = 1,
    CUBIC     = 2,

    INVALID = -1
};

enum class covariance_algorithm_enum : int16_t
{
    DENSE_SVD = 0,
    SPARSE_QR = 1,

    INVALID = -1
};

enum class dump_format_enum : int16_t
{
    // Print the linear least squares problem in a human readable format
    // to stderr. The Jacobian is printed as a dense matrix. The vectors
    // D, x and f are printed as dense vectors. This should only be used
    // for small problems.
    CONSOLE = 0,

    // Write out the linear least squares problem to the directory
    // pointed to by Solver::Options::lsqp_dump_directory as text files
    // which can be read into MATLAB/Octave. The Jacobian is dumped as a
    // text file containing (i,j,s) triplets, the vectors D, x and f are
    // dumped as text files containing a list of their values.
    //
    // A MATLAB/octave script called lm_iteration_???.m is also output,
    // which can be used to parse and load the problem into memory.
    TEXTFILE = 1,

    INVALID = -1
};
}  // namespace quarisma

namespace quarisma
{
class MATH_VISIBILITY solver_options_ceres : public solver_options
{
    friend class solver_options_ceres_builder;

public:
    solver_options_ceres(
        int    max_num_iterations,
        double function_tolerance  = std::numeric_limits<double>::epsilon(),
        double gradient_tolerance  = std::numeric_limits<double>::epsilon(),
        double parameter_tolerance = std::numeric_limits<double>::epsilon(),
        bool   verbose             = false)
        : solver_options(
              solver_enum::CERES,
              max_num_iterations,
              function_tolerance,
              gradient_tolerance,
              parameter_tolerance,
              verbose)
    {
    }

    // Getters and Setters
    visibility_clustering_enum visibility_clustering_type() const
    {
        return visibility_clustering_type_;
    }
    void set_visibility_clustering_type(visibility_clustering_enum type)
    {
        visibility_clustering_type_ = type;
    }

    preconditioner_enum preconditioner_type() const { return preconditioner_type_; }
    void set_preconditioner_type(preconditioner_enum type) { preconditioner_type_ = type; }

    sparse_linear_algebra_library_enum sparse_linear_algebra_library_type() const
    {
        return sparse_linear_algebra_library_type_;
    }
    void set_sparse_linear_algebra_library_type(sparse_linear_algebra_library_enum type)
    {
        sparse_linear_algebra_library_type_ = type;
    }

    dense_linear_algebra_library_enum dense_linear_algebra_library_type() const
    {
        return dense_linear_algebra_library_type_;
    }
    void set_dense_linear_algebra_library_type(dense_linear_algebra_library_enum type)
    {
        dense_linear_algebra_library_type_ = type;
    }

    linear_solver_enum linear_solver_type() const { return linear_solver_type_; }
    void set_linear_solver_type(linear_solver_enum type) { linear_solver_type_ = type; }

    linear_solver_ordering_enum linear_solver_ordering_type() const
    {
        return linear_solver_ordering_type_;
    }
    void set_linear_solver_ordering_type(linear_solver_ordering_enum type)
    {
        linear_solver_ordering_type_ = type;
    }

    dogleg_enum dogleg_type() const { return dogleg_type_; }
    void        set_dogleg_type(dogleg_enum type) { dogleg_type_ = type; }

    trust_region_strategy_enum trust_region_strategy_type() const
    {
        return trust_region_strategy_type_;
    }
    void set_trust_region_strategy_type(trust_region_strategy_enum type)
    {
        trust_region_strategy_type_ = type;
    }

    line_search_interpolation_enum line_search_interpolation_type() const
    {
        return line_search_interpolation_type_;
    }
    void set_line_search_interpolation_type(line_search_interpolation_enum type)
    {
        line_search_interpolation_type_ = type;
    }

    minimizer_enum minimizer_type() const { return minimizer_type_; }
    void           set_minimizer_type(minimizer_enum type) { minimizer_type_ = type; }

    line_search_direction_enum line_search_direction_type() const
    {
        return line_search_direction_type_;
    }
    void set_line_search_direction_type(line_search_direction_enum type)
    {
        line_search_direction_type_ = type;
    }

    line_search_enum line_search_type() const { return line_search_type_; }
    void             set_line_search_type(line_search_enum type) { line_search_type_ = type; }

    nonlinear_conjugate_gradient_enum nonlinear_conjugate_gradient_type() const
    {
        return nonlinear_conjugate_gradient_type_;
    }
    void set_nonlinear_conjugate_gradient_type(nonlinear_conjugate_gradient_enum type)
    {
        nonlinear_conjugate_gradient_type_ = type;
    }

    numeric_diff_method_enum numeric_diff_method() const { return numeric_diff_method_; }
    void set_numeric_diff_method(numeric_diff_method_enum type) { numeric_diff_method_ = type; }

    int  max_lbfgs_rank() const { return max_lbfgs_rank_; }
    void set_max_lbfgs_rank(int rank) { max_lbfgs_rank_ = rank; }

    double initial_trust_region_radius() const { return initial_trust_region_radius_; }
    void   set_initial_trust_region_radius(double radius) { initial_trust_region_radius_ = radius; }

    double max_trust_region_radius() const { return max_trust_region_radius_; }
    void   set_max_trust_region_radius(double radius) { max_trust_region_radius_ = radius; }

    double min_trust_region_radius() const { return min_trust_region_radius_; }
    void   set_min_trust_region_radius(double radius) { min_trust_region_radius_ = radius; }

    double min_relative_decrease() const { return min_relative_decrease_; }
    void   set_min_relative_decrease(double decrease) { min_relative_decrease_ = decrease; }

    double eta() const { return eta_; }
    void   set_eta(double eta) { eta_ = eta; }

    bool use_inner_iterations() const { return use_inner_iterations_; }
    void set_use_inner_iterations(bool use) { use_inner_iterations_ = use; }

    double inner_iteration_tolerance() const { return inner_iteration_tolerance_; }
    void set_inner_iteration_tolerance(double tolerance) { inner_iteration_tolerance_ = tolerance; }

    bool jacobi_scaling() const { return jacobi_scaling_; }
    void set_jacobi_scaling(bool scaling) { jacobi_scaling_ = scaling; }

    bool dynamic_sparsity() const { return dynamic_sparsity_; }
    void set_dynamic_sparsity(bool sparsity) { dynamic_sparsity_ = sparsity; }

    bool use_mixed_precision_solves() const { return use_mixed_precision_solves_; }
    void set_use_mixed_precision_solves(bool use) { use_mixed_precision_solves_ = use; }

    int  max_num_refinement_iterations() const { return max_num_refinement_iterations_; }
    void set_max_num_refinement_iterations(int iterations)
    {
        max_num_refinement_iterations_ = iterations;
    }

    double min_line_search_step_size() const { return min_line_search_step_size_; }
    void set_min_line_search_step_size(double step_size) { min_line_search_step_size_ = step_size; }

    double line_search_sufficient_function_decrease() const
    {
        return line_search_sufficient_function_decrease_;
    }
    void set_line_search_sufficient_function_decrease(double decrease)
    {
        line_search_sufficient_function_decrease_ = decrease;
    }

    double line_search_sufficient_curvature_decrease() const
    {
        return line_search_sufficient_curvature_decrease_;
    }
    void set_line_search_sufficient_curvature_decrease(double decrease)
    {
        line_search_sufficient_curvature_decrease_ = decrease;
    }

    double max_line_search_step_contraction() const { return max_line_search_step_contraction_; }
    void   set_max_line_search_step_contraction(double contraction)
    {
        max_line_search_step_contraction_ = contraction;
    }

    double min_line_search_step_contraction() const { return min_line_search_step_contraction_; }
    void   set_min_line_search_step_contraction(double contraction)
    {
        min_line_search_step_contraction_ = contraction;
    }

    double max_line_search_step_expansion() const { return max_line_search_step_expansion_; }
    void   set_max_line_search_step_expansion(double expansion)
    {
        max_line_search_step_expansion_ = expansion;
    }

    bool check_gradients() const { return check_gradients_; }
    void set_check_gradients(bool check) { check_gradients_ = check; }

    double gradient_check_relative_precision() const { return gradient_check_relative_precision_; }
    void   set_gradient_check_relative_precision(double precision)
    {
        gradient_check_relative_precision_ = precision;
    }

    double gradient_check_numeric_derivative_relative_step_size() const
    {
        return gradient_check_numeric_derivative_relative_step_size_;
    }
    void set_gradient_check_numeric_derivative_relative_step_size(double step_size)
    {
        gradient_check_numeric_derivative_relative_step_size_ = step_size;
    }

    double max_solver_time_in_seconds() const { return max_solver_time_in_seconds_; }
    void   set_max_solver_time_in_seconds(double time) { max_solver_time_in_seconds_ = time; }

    int  num_threads() const { return num_threads_; }
    void set_num_threads(int threads) { num_threads_ = threads; }

    int  min_linear_solver_iterations() const { return min_linear_solver_iterations_; }
    void set_min_linear_solver_iterations(int iterations)
    {
        min_linear_solver_iterations_ = iterations;
    }

    int  max_linear_solver_iterations() const { return max_linear_solver_iterations_; }
    void set_max_linear_solver_iterations(int iterations)
    {
        max_linear_solver_iterations_ = iterations;
    }

    const std::vector<int>& trust_region_minimizer_iterations_to_dump() const
    {
        return trust_region_minimizer_iterations_to_dump_;
    }
    void set_trust_region_minimizer_iterations_to_dump(const std::vector<int>& iterations)
    {
        trust_region_minimizer_iterations_to_dump_ = iterations;
    }

    const std::string& trust_region_problem_dump_directory() const
    {
        return trust_region_problem_dump_directory_;
    }
    void set_trust_region_problem_dump_directory(const std::string& directory)
    {
        trust_region_problem_dump_directory_ = directory;
    }

    dump_format_enum trust_region_problem_dump_format_type() const
    {
        return trust_region_problem_dump_format_type_;
    }
    void set_trust_region_problem_dump_format_type(dump_format_enum type)
    {
        trust_region_problem_dump_format_type_ = type;
    }

private:
    void initialize() const {};

    solver_options_ceres() : solver_options(quarisma::solver_enum::CERES) {};

    QUARISMA_SERIALIZATION_EXPORT(
        MATH_API,
        solver_options_ceres,
        solver_,
        max_num_iterations_,
        function_tolerance_,
        gradient_tolerance_,
        parameter_tolerance_,
        verbose_,
        aad_jacobian_,
        log_file_,
        visibility_clustering_type_,
        preconditioner_type_,
        sparse_linear_algebra_library_type_,
        dense_linear_algebra_library_type_,
        linear_solver_type_,
        linear_solver_ordering_type_,
        dogleg_type_,
        trust_region_strategy_type_,
        line_search_interpolation_type_,
        minimizer_type_,
        line_search_direction_type_,
        line_search_type_,
        nonlinear_conjugate_gradient_type_,
        numeric_diff_method_,
        max_lbfgs_rank_,
        initial_trust_region_radius_,
        max_trust_region_radius_,
        min_trust_region_radius_,
        min_relative_decrease_,
        eta_,
        use_inner_iterations_,
        inner_iteration_tolerance_,
        jacobi_scaling_,
        dynamic_sparsity_,
        use_mixed_precision_solves_,
        max_num_refinement_iterations_,
        min_line_search_step_size_,
        line_search_sufficient_function_decrease_,
        line_search_sufficient_curvature_decrease_,
        max_line_search_step_contraction_,
        min_line_search_step_contraction_,
        max_line_search_step_expansion_,
        check_gradients_,
        gradient_check_relative_precision_,
        gradient_check_numeric_derivative_relative_step_size_,
        max_solver_time_in_seconds_,
        num_threads_,
        min_linear_solver_iterations_,
        max_linear_solver_iterations_,
        trust_region_minimizer_iterations_to_dump_,
        trust_region_problem_dump_directory_,
        trust_region_problem_dump_format_type_);

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

/**
 * @brief Builder for solver_options_ceres configuration
 *
 * Usage:
 *   auto options = solver_options_ceres_builder()
 *       .with_max_iterations(100).with_function_tolerance(1e-6)
 *       .with_minimizer_type(MinimizerType::TRUST_REGION)
 *       .with_linear_solver_type(LinearSolverType::DENSE_QR).build();
 */
class MATH_VISIBILITY solver_options_ceres_builder
{
public:
    solver_options_ceres_builder()
        : options_(ptr_mutable<solver_options_ceres>(new solver_options_ceres()))
    {
    }

    // Base solver options
    quarisma::solver_options_ceres_builder& with_max_iterations(int val)
    {
        options_->set_max_num_iterations(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_function_tolerance(double val)
    {
        options_->set_function_tolerance(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_gradient_tolerance(double val)
    {
        options_->set_gradient_tolerance(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_parameter_tolerance(double val)
    {
        options_->set_parameter_tolerance(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_verbose(bool val = true)
    {
        options_->set_debug(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_log_file(const std::string& val)
    {
        options_->log_file_ = val;
        return *this;
    }

    // Core Ceres options
    quarisma::solver_options_ceres_builder& with_linear_solver_type(linear_solver_enum val)
    {
        options_->set_linear_solver_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_preconditioner_type(preconditioner_enum val)
    {
        options_->set_preconditioner_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_minimizer_type(minimizer_enum val)
    {
        options_->set_minimizer_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_trust_region_strategy_type(
        trust_region_strategy_enum val)
    {
        options_->set_trust_region_strategy_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_num_threads(int val)
    {
        options_->set_num_threads(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_max_solver_time_in_seconds(double val)
    {
        options_->set_max_solver_time_in_seconds(val);
        return *this;
    }

    // Additional enum class builder methods
    quarisma::solver_options_ceres_builder& with_dogleg_type(dogleg_enum val)
    {
        options_->set_dogleg_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_line_search_direction_type(
        line_search_direction_enum val)
    {
        options_->set_line_search_direction_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_line_search_type(line_search_enum val)
    {
        options_->set_line_search_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_nonlinear_conjugate_gradient_type(
        nonlinear_conjugate_gradient_enum val)
    {
        options_->set_nonlinear_conjugate_gradient_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_line_search_interpolation_type(
        line_search_interpolation_enum val)
    {
        options_->set_line_search_interpolation_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_numeric_diff_method(numeric_diff_method_enum val)
    {
        options_->set_numeric_diff_method(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_sparse_linear_algebra_library_type(
        sparse_linear_algebra_library_enum val)
    {
        options_->set_sparse_linear_algebra_library_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_dense_linear_algebra_library_type(
        dense_linear_algebra_library_enum val)
    {
        options_->set_dense_linear_algebra_library_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_linear_solver_ordering_type(
        linear_solver_ordering_enum val)
    {
        options_->set_linear_solver_ordering_type(val);
        return *this;
    }

    quarisma::solver_options_ceres_builder& with_aad_jacobian(bool val = true)
    {
        options_->aad_jacobian_ = val;
        return *this;
    }

    // Build the final options
    ptr_const<solver_options_ceres> build() const { return options_; }

private:
    ptr_mutable<solver_options_ceres> options_;
};

}  // namespace quarisma
