#ifndef SOLVERS_EIGEN_SUPPORT_H_
#define SOLVERS_EIGEN_SUPPORT_H_
// Every third-party linear-algebra dependency for this library is confined
// to this single header. Solver headers and implementation files only ever
// see solverslib::vector_type / solverslib::matrix_type and the free
// functions declared below - never an "Eigen::"-qualified name or an
// <Eigen/...> include of their own. Swapping the dense linear-algebra
// backend later means editing only this file (and its call sites here),
// not every solver.
//
// This mirrors how Eigen and PyTorch structure their own APIs: Eigen's
// Matrix/Array expression templates and PyTorch's ATen Tensor are both
// opaque value types whose *methods and operators* form the public
// numerical surface, while backend/storage specifics (interop with raw
// buffers, dense decompositions, row-major/column-major layout) stay behind
// free functions or small wrapper classes the caller never names directly.
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include <cstddef>
#include <vector>

namespace solverslib
{
using vector_type = Eigen::VectorXd;
using matrix_type = Eigen::MatrixXd;
using index_type  = Eigen::Index;

// -- construction -----------------------------------------------------------
inline vector_type make_vector(size_t size)
{
    return vector_type(static_cast<index_type>(size));
}

inline matrix_type make_matrix(size_t rows, size_t cols)
{
    return matrix_type(static_cast<index_type>(rows), static_cast<index_type>(cols));
}

// -- sizing -------------------------------------------------------------------
// Resize only when the shape differs, so repeated calls on a correctly sized
// buffer never touch the allocator.
inline void resize_if_needed(vector_type& v, size_t size)
{
    if (static_cast<size_t>(v.size()) != size)
    {
        v.resize(static_cast<index_type>(size));
    }
}

inline void resize_if_needed(matrix_type& a, size_t rows, size_t cols)
{
    if (static_cast<size_t>(a.rows()) != rows || static_cast<size_t>(a.cols()) != cols)
    {
        a.resize(static_cast<index_type>(rows), static_cast<index_type>(cols));
    }
}

// -- interop with raw buffers / std::vector<double> --------------------------
// The only place this library touches a bare double* or std::vector<double>
// boundary: Ceres' C-style API, and the solve() dispatcher's
// std::vector<double> parameter storage.
inline vector_type to_vector_type(const double* data, size_t size)
{
    return Eigen::Map<const vector_type>(data, static_cast<index_type>(size));
}

inline vector_type to_vector_type(const std::vector<double>& values)
{
    return to_vector_type(values.data(), values.size());
}

inline void copy_into(double* destination, size_t size, const vector_type& source)
{
    Eigen::Map<vector_type>(destination, static_cast<index_type>(size)) = source;
}

inline void copy_into(std::vector<double>& destination, const vector_type& source)
{
    copy_into(destination.data(), destination.size(), source);
}

// Fill `destination` (resized to rows x cols) from a flat row-major buffer, the
// layout Ceres writes Jacobians in.
inline void copy_from_row_major(
    matrix_type& destination, const double* source, size_t rows, size_t cols)
{
    resize_if_needed(destination, rows, cols);
    destination =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            source, static_cast<index_type>(rows), static_cast<index_type>(cols));
}

// Test hook: forbid (or allow again) heap allocation by the linear-algebra
// backend. Active only when the backend's runtime allocation guard is compiled
// in (Eigen: EIGEN_RUNTIME_NO_MALLOC); otherwise a no-op.
inline void set_allocation_allowed(bool allowed)
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(allowed);
#else
    (void)allowed;
#endif
}

// Ceres' CostFunction::Evaluate hands back jacobians as a flat row-major
// double*; this is the one place that layout detail needs to be named.
inline void copy_row_major(double* destination, const matrix_type& source)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        destination, source.rows(), source.cols()) = source;
}

// -- linear solves ------------------------------------------------------------
// Wraps the one dense decomposition this library needs (the LM
// normal-equation step). Keeping the factorization object alive across
// solve() calls (as levenberg_marquardt_solver does for its geodesic
// acceleration correction) avoids refactorizing the same matrix twice.
// solve() returns a fully materialized vector_type rather than a lazy
// expression, so it is always safe to assign into one of its own inputs.
class linear_system_solver
{
public:
    explicit linear_system_solver(const matrix_type& a) : factorization_(a) {}

    vector_type solve(const vector_type& b) const { return factorization_.solve(b); }

private:
    Eigen::PartialPivLU<matrix_type> factorization_;
};

// -- damped least-squares step (Levenberg-Marquardt) ------------------------------
// Solves  (J^T J + diag(damping)) * delta = J^T y  for the LM step, in one of
// two ways. Both are reached through this class, so solver code keeps a single
// dense-algebra boundary.
//   normal_ldlt   LDLT of the normal-equations matrix; fastest for m >> n.
//   augmented_qr  QR of [J; diag(sqrt(damping))], solving the same system as the
//                 least-squares problem  min || [J; sqrt(D)] delta - [y; 0] ||.
//                 It never forms J^T J, so the condition number is not squared.
// Eigen's own unsupported LevenbergMarquardt module and MINPACK lmder use the
// QR form; Ceres offers both for the same reason. Buffers are sized once at
// construction so refactoring each iteration does not allocate (normal_ldlt).
enum class damped_step_method : int
{
    normal_ldlt  = 0,
    augmented_qr = 1
};

class damped_step_solver
{
public:
    damped_step_solver(index_type rows, index_type cols, damped_step_method method)
        : method_(method), rows_(rows), cols_(cols),
          ldlt_(method == damped_step_method::normal_ldlt ? cols : 0),
          qr_(method == damped_step_method::augmented_qr ? rows + cols : 0,
              method == damped_step_method::augmented_qr ? cols : 0),
          damped_(method == damped_step_method::normal_ldlt ? cols : 0,
              method == damped_step_method::normal_ldlt ? cols : 0),
          augmented_(method == damped_step_method::augmented_qr ? rows + cols : 0,
              method == damped_step_method::augmented_qr ? cols : 0),
          rhs_(method == damped_step_method::augmented_qr ? rows + cols : 0)
    {
    }

    // `jtj` is J^T J (used by normal_ldlt); `damping` is the diagonal added to it.
    // Returns false when the factorization breaks down (the caller should raise
    // the damping and retry).
    bool factor(const matrix_type& jacobian, const matrix_type& jtj, const vector_type& damping)
    {
        if (method_ == damped_step_method::normal_ldlt)
        {
            damped_ = jtj;
            damped_.diagonal() += damping;
            ok_ = damped_.allFinite();
            if (ok_)
            {
                ldlt_.compute(damped_);
                ok_ = ldlt_.info() == Eigen::Success && ldlt_.isPositive() &&
                      ldlt_.vectorD().allFinite();
            }
        }
        else
        {
            augmented_.topRows(rows_) = jacobian;
            augmented_.bottomRows(cols_).setZero();
            augmented_.bottomRows(cols_).diagonal() = damping.cwiseMax(0.).cwiseSqrt();
            ok_                                     = augmented_.allFinite();
            if (ok_)
            {
                qr_.compute(augmented_);
                ok_ = qr_.info() == Eigen::Success && qr_.rank() == cols_;
            }
        }
        return ok_;
    }

    // delta for the right-hand side y in residual space; `jt_y` must equal J^T y
    // (supplied by the caller, who usually has it already). Valid after a
    // successful factor().
    void solve(const vector_type& y, const vector_type& jt_y, vector_type& delta)
    {
        if (method_ == damped_step_method::normal_ldlt)
        {
            delta = ldlt_.solve(jt_y);
        }
        else
        {
            rhs_.head(rows_) = y;
            rhs_.tail(cols_).setZero();
            delta = qr_.solve(rhs_);
        }
    }

private:
    damped_step_method                      method_;
    index_type                              rows_;
    index_type                              cols_;
    Eigen::LDLT<matrix_type>                ldlt_;
    Eigen::ColPivHouseholderQR<matrix_type> qr_;
    matrix_type                             damped_;
    matrix_type                             augmented_;
    vector_type                             rhs_;
    bool                                    ok_ = false;
};

// RNC coefficient systems share one symmetric positive-definite matrix.
class positive_definite_solver
{
public:
    explicit positive_definite_solver(const matrix_type& a) : factorization_(a) {}
    bool        valid() const { return factorization_.info() == Eigen::Success; }
    vector_type solve(const vector_type& b) const { return factorization_.solve(b); }

private:
    Eigen::LLT<matrix_type> factorization_;
};

}  // namespace solverslib

#endif  // SOLVERS_EIGEN_SUPPORT_H_
