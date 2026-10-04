#ifndef SOLVERS_EVALUATOR_H_
#define SOLVERS_EVALUATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "detail/eigen_support.h"
#include "solvers/api/status.h"

// Design note: Ceres separates Problem (what the user declared) from Evaluator
// (a per-solve object that evaluates residuals and an optional Jacobian into
// buffers it owns). PyTorch's autograd.Function likewise keeps per-call state
// in a context rather than recomputing. These evaluators have that shape, with
// dense storage and a single parameter block.
namespace solverslib::api::detail
{
// Evaluation result codes
enum class evaluation_status : std::uint8_t
{
    ok,             // successful evaluation
    invalid_trial,  // evaluation succeeded but returned false (e.g., out of domain)
    fatal_error     // exception or NaN/Inf in output
};

// Immutable metadata about a provider
struct provider_metadata
{
    std::size_t          num_parameters;
    std::size_t          num_residuals;
    api::derivative_mode source;  // supplied, automatic_differentiation, or finite_difference
    bool                 supports_ceres = false;
};

// Work done by one evaluator. `residual_evaluations` counts every execution of
// the user's residual function, including those made inside a finite-difference
// stencil or alongside an analytic Jacobian.
struct evaluation_counters
{
    std::size_t residual_evaluations = 0;
    std::size_t jacobian_evaluations = 0;
};

// Abstract per-solve evaluator: residuals and optional Jacobians. Not
// thread-safe; a solve owns one. The public entry points are non-virtual so the
// counters cannot be skipped by an implementation.
class residual_evaluator
{
public:
    virtual ~residual_evaluator() = default;

    // Compute residuals and optionally the Jacobian at x.
    // - residuals: output, resized to num_residuals when needed
    // - jacobians: m x n output, or nullptr for residuals only
    evaluation_status evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians = nullptr)
    {
        ++counters_.residual_evaluations;
        if (jacobians != nullptr)
        {
            ++counters_.jacobian_evaluations;
        }
        return do_evaluate(x, residuals, jacobians);
    }

    // Jacobian at a point whose residuals the caller already holds. Lets
    // callback and finite-difference evaluators skip re-evaluating residuals.
    evaluation_status jacobian(
        const vector_type& x, const vector_type& residuals_at_x, matrix_type& jacobian_out)
    {
        ++counters_.jacobian_evaluations;
        return do_jacobian(x, residuals_at_x, jacobian_out);
    }

    virtual const provider_metadata& metadata() const = 0;

    // For fatal errors: exception message or NaN/Inf details
    virtual std::optional<std::string> last_error() const { return std::nullopt; }

    const evaluation_counters& counters() const noexcept { return counters_; }

protected:
    virtual evaluation_status do_evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians) = 0;

    // Default: evaluate both and discard the residuals.
    virtual evaluation_status do_jacobian(
        const vector_type& x, const vector_type& /*residuals_at_x*/, matrix_type& jacobian_out)
    {
        vector_type scratch = make_vector(metadata().num_residuals);
        ++counters_.residual_evaluations;
        return do_evaluate(x, scratch, &jacobian_out);
    }

    evaluation_counters counters_;
};

// Work done by one scalar-objective evaluator.
struct gradient_counters
{
    std::size_t objective_evaluations = 0;
    std::size_t gradient_evaluations  = 0;
};

// Scalar-objective counterpart of residual_evaluator.
class gradient_evaluator
{
public:
    virtual ~gradient_evaluator() = default;

    // Objective value at x, and optionally its gradient.
    evaluation_status evaluate(const vector_type& x, double& value, vector_type* gradient = nullptr)
    {
        ++counters_.objective_evaluations;
        if (gradient != nullptr)
        {
            ++counters_.gradient_evaluations;
        }
        return do_evaluate(x, value, gradient);
    }

    // Gradient at a point whose value the caller already holds.
    evaluation_status gradient(const vector_type& x, vector_type& gradient_out)
    {
        ++counters_.gradient_evaluations;
        return do_gradient(x, gradient_out);
    }

    virtual std::size_t          num_parameters() const = 0;
    virtual api::derivative_mode source() const         = 0;

    virtual std::optional<std::string> last_error() const { return std::nullopt; }

    const gradient_counters& counters() const noexcept { return counters_; }

protected:
    virtual evaluation_status do_evaluate(
        const vector_type& x, double& value, vector_type* gradient) = 0;

    // Default: evaluate both and discard the value.
    virtual evaluation_status do_gradient(const vector_type& x, vector_type& gradient_out)
    {
        double scratch = 0.0;
        ++counters_.objective_evaluations;
        return do_evaluate(x, scratch, &gradient_out);
    }

    gradient_counters counters_;
};

// Factory: produces one evaluator per solve (owned by that solve)
class provider_factory
{
public:
    virtual ~provider_factory() = default;

    virtual const provider_metadata& metadata() const = 0;

    // Create an evaluator for one solve
    // Must be thread-safe per call (evaluator ownership is per-thread after creation)
    virtual std::unique_ptr<residual_evaluator> create_evaluator() const = 0;
};

}  // namespace solverslib::api::detail

#endif  // SOLVERS_EVALUATOR_H_
