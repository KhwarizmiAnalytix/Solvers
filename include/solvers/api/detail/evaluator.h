#ifndef SOLVERS_EVALUATOR_H_
#define SOLVERS_EVALUATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "detail/eigen_support.h"
#include "solvers/api/status.h"

namespace solverslib::api::detail
{
// Evaluation result codes
enum class evaluation_status
{
    ok,            // successful evaluation
    invalid_trial, // evaluation succeeded but returned false (e.g., out of domain)
    fatal_error    // exception or NaN/Inf in output
};

// Immutable metadata about a provider
struct provider_metadata
{
    std::size_t num_parameters;
    std::size_t num_residuals;
    api::derivative_mode source;  // supplied, automatic_differentiation, or finite_difference
    bool supports_ceres = false;
};

// Abstract evaluator: compute residuals and optionally Jacobians
class residual_evaluator
{
public:
    virtual ~residual_evaluator() = default;

    // Compute residuals and optionally Jacobians
    // - x: parameter values
    // - residuals: output buffer, size >= num_residuals
    // - jacobians: pointer to row-major Jacobian (m rows x n cols), or nullptr for residuals-only
    // Returns evaluation_status; stores fatal errors for later retrieval
    virtual evaluation_status evaluate(
        const vector_type& x,
        vector_type& residuals,
        matrix_type* jacobians = nullptr) = 0;

    virtual const provider_metadata& metadata() const = 0;

    // For fatal errors: exception message or NaN/Inf details
    virtual std::optional<std::string> last_error() const { return std::nullopt; }
};

// Factory: produces one evaluator per solve (owned by that solve)
class provider_factory
{
public:
    virtual ~provider_factory() = default;

    virtual const provider_metadata& metadata() const = 0;

    // Create an evaluator for one solve
    // Must be thread-safe per call (evaluator ownership is per-thread after creation)
    virtual std::unique_ptr<residual_evaluator> create_evaluator() = 0;
};

}  // namespace solverslib::api::detail

#endif  // SOLVERS_EVALUATOR_H_
