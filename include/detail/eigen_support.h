#pragma once
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
#include <Eigen/Core>
#include <Eigen/LU>

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

// -- interop with raw buffers / std::vector<double> --------------------------
// The only place this library touches a bare double* or std::vector<double>
// boundary: Ceres' and NLopt's C-style APIs, and solver_wrapper's
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

}  // namespace solverslib
