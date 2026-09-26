#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "include/detail/eigen_support.h"
#include "include/detail/support.h"

namespace solverslib::api
{
// Callback vocabulary. Residual/Jacobian follow the existing native solver
// signatures so the native backend can adopt them without copies. The scalar
// objective/gradient/Hessian family describes general optimization; it stays a
// distinct contract from residual least squares (review: "Keep general scalar
// objectives separate").
using residual_function = std::function<void(const vector_type&, vector_type&)>;
using jacobian_function = std::function<void(const vector_type&, matrix_type&)>;

using objective_function = std::function<double(const vector_type&)>;
using gradient_function  = std::function<void(const vector_type&, vector_type&)>;
using hessian_function   = std::function<void(const vector_type&, matrix_type&)>;
// Matrix-free Hessian-vector product H(x) * v -> out. Its presence steers the
// dispatcher toward a matrix-free (TAO/Newton-Krylov) backend even below the
// size thresholds (review section 7).
using hessian_vector_function =
    std::function<void(const vector_type& x, const vector_type& v, vector_type& out)>;

// Optional box constraints. Either side may be provided independently; an empty
// vector means "no bound on that side" (review F01: one-sided bounds must not be
// silently discarded).
struct bounds
{
    std::vector<double> lower;
    std::vector<double> upper;

    bool empty() const noexcept { return lower.empty() && upper.empty(); }
    bool has_lower() const noexcept { return !lower.empty(); }
    bool has_upper() const noexcept { return !upper.empty(); }
};

// General nonlinear constraints. Only their presence is modeled here; the first
// implementation slice rejects them on backends that cannot enforce them.
struct constraints
{
    std::size_t num_equality   = 0;
    std::size_t num_inequality = 0;

    bool empty() const noexcept { return num_equality == 0 && num_inequality == 0; }
};

// F(x) = 0.5 * ||r(x)||^2, J(i,j) = d r_i / d x_j, g = J^T r.
struct least_squares_problem
{
    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;

    residual_function                residuals;
    std::optional<jacobian_function> jacobian;

    api::bounds bounds;
};

// General objective min f(x).
struct optimization_problem
{
    std::size_t num_parameters = 0;

    objective_function                     objective;
    std::optional<gradient_function>       gradient;
    std::optional<hessian_function>        hessian;
    std::optional<hessian_vector_function> hessian_vector;

    api::bounds      bounds;
    api::constraints constraints;
};

// Structural summary derived from a problem. Drives backend selection so the
// backend never has to be the primary abstraction (review section 2).
struct problem_traits
{
    bool is_least_squares           = false;
    bool has_jacobian               = false;
    bool has_gradient               = false;
    bool has_hessian                = false;
    bool has_hessian_vector_product = false;
    bool has_bounds                 = false;
    bool has_nonlinear_constraints  = false;

    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;
};

// One explicit policy for "large scale" rather than magic thresholds scattered
// through the dispatcher (review section 7).
struct dispatch_policy
{
    std::size_t large_parameter_threshold = 1000;
    std::size_t large_residual_threshold  = 10000;
    bool        prefer_matrix_free        = true;
};
}  // namespace solverslib::api
