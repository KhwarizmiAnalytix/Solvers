#ifndef SOLVERS_PROBLEM_H_
#define SOLVERS_PROBLEM_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solvers/api/derivative_provider.h"
#include "solvers/api/status.h"

namespace solverslib::api
{
// Callback vocabulary. Residual/Jacobian follow the existing native solver
// signatures so the native backend can adopt them without copies. The scalar
// objective/gradient/Hessian family describes general optimization; it stays a
// distinct contract from residual least squares.
using residual_function = std::function<void(const vector_type&, vector_type&)>;
using jacobian_function = std::function<void(const vector_type&, matrix_type&)>;

using objective_function = std::function<double(const vector_type&)>;
using gradient_function  = std::function<void(const vector_type&, vector_type&)>;
using hessian_function   = std::function<void(const vector_type&, matrix_type&)>;
// Matrix-free Hessian-vector product H(x) * v -> out.
using hessian_vector_function =
    std::function<void(const vector_type& x, const vector_type& v, vector_type& out)>;

// Optional box constraints. Either side may be provided independently; an empty
// vector means "no bound on that side".
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
//
// Plain aggregate: no setters or factory logic. All fields are public.
// Derivatives: supply jacobian (e.g., from analytic code or finite differences),
// or leave it null (finite differences will be applied automatically if needed).
// For RNC-LM, optionally supply curve_derivatives.
struct least_squares_problem
{
    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;

    residual_function residuals;

    // Optional: Jacobian callback. If absent, finite differences will be used.
    std::optional<jacobian_function> jacobian;

    // Optional: Derivatives along a curve for RNC-LM (analytic or Taylor AD).
    // Order 1 with empty coefficients gives residual and Jacobian at base point.
    std::optional<rnc_derivative_function> curve_derivatives;

    api::bounds bounds;
};

// General objective min f(x). One gradient slot.
struct optimization_problem
{
    std::size_t num_parameters = 0;

    objective_function                     objective;
    std::optional<hessian_function>        hessian;
    std::optional<hessian_vector_function> hessian_vector;

    // Optional: gradient callback. If absent, finite differences will be used.
    std::optional<gradient_function> gradient;

    api::bounds      bounds;
    api::constraints constraints;
};

// Structural summary derived from a problem. Drives backend selection.
struct problem_traits
{
    bool is_least_squares           = false;
    bool has_jacobian               = false;  // any Jacobian source
    bool has_gradient               = false;  // any gradient source
    bool has_hessian                = false;
    bool has_hessian_vector_product = false;
    bool has_bounds                 = false;
    bool has_nonlinear_constraints  = false;

    std::size_t num_parameters = 0;
    std::size_t num_residuals  = 0;
};

}  // namespace solverslib::api

#endif  // SOLVERS_PROBLEM_H_
