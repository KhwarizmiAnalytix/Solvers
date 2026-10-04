#ifndef SOLVERS_EVALUATORS_H_
#define SOLVERS_EVALUATORS_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "detail/eigen_support.h"
#include "solvers/api/detail/evaluator.h"
#include "solvers/api/options.h"
#include "solvers/api/problem.h"
#include "solvers/api/result.h"

// Internal: the concrete per-solve evaluators and the one derivative-policy
// resolver every backend goes through. Not part of the stable API.
namespace solverslib::api::detail
{
// Default relative finite-difference step. Central differences have their
// error optimum near eps^(1/3); 1e-5 matches what the native kernels used.
inline constexpr double default_fd_step = 1e-5;

// Scratch for the finite-difference stencils; evaluators own one so repeated
// Jacobians do not allocate.
struct finite_difference_workspace
{
    vector_type probe;
    vector_type plus;
    vector_type minus;
};

// -- the single finite-difference implementation -------------------------------
// h_j = relative_step * max(1, |x_j|). Central stencil inside the bounds; a
// one-sided stencil toward the interior when either side would leave them.
// `evaluations` is incremented once per call of `f`.
void finite_difference_jacobian(const residual_function& f,
    const vector_type&                                   x,
    const vector_type&                                   residuals_at_x,
    double                                               relative_step,
    const api::bounds&                                   bounds,
    matrix_type&                                         jacobian,
    std::size_t&                                         evaluations,
    finite_difference_workspace*                         workspace = nullptr);

// `value_at_x` may be null; it is then computed only if a one-sided stencil needs it.
void finite_difference_gradient(const objective_function& f,
    const vector_type&                                    x,
    const double*                                         value_at_x,
    double                                                relative_step,
    const api::bounds&                                    bounds,
    vector_type&                                          gradient,
    std::size_t&                                          evaluations,
    finite_difference_workspace*                          workspace = nullptr);

// -- residual evaluators ---------------------------------------------------------
// Residuals from a callback; Jacobian from a callback (when given).
class callback_residual_evaluator final : public residual_evaluator
{
public:
    callback_residual_evaluator(std::size_t n,
        std::size_t                         m,
        residual_function                   residuals,
        std::optional<jacobian_function>    jacobian = std::nullopt);

    const provider_metadata&   metadata() const override { return metadata_; }
    std::optional<std::string> last_error() const override { return last_error_; }

protected:
    evaluation_status do_evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians) override;
    evaluation_status do_jacobian(
        const vector_type& x, const vector_type& residuals_at_x, matrix_type& jacobian) override;

private:
    residual_function                residuals_;
    std::optional<jacobian_function> jacobian_;
    provider_metadata                metadata_;
    std::optional<std::string>       last_error_;
};

// Residuals from a callback; Jacobian by finite differences of that callback.
class finite_difference_residual_evaluator final : public residual_evaluator
{
public:
    finite_difference_residual_evaluator(std::size_t n,
        std::size_t                                  m,
        residual_function                            residuals,
        double                                       relative_step = default_fd_step,
        api::bounds                                  bounds        = {});

    const provider_metadata&   metadata() const override { return metadata_; }
    std::optional<std::string> last_error() const override { return last_error_; }

protected:
    evaluation_status do_evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians) override;
    evaluation_status do_jacobian(
        const vector_type& x, const vector_type& residuals_at_x, matrix_type& jacobian) override;

private:
    residual_function           residuals_;
    double                      relative_step_;
    api::bounds                 bounds_;
    provider_metadata           metadata_;
    std::optional<std::string>  last_error_;
    finite_difference_workspace workspace_;
};

// Residuals from the user's callback; derivatives from another evaluator
// (an analytic/AD provider adapter or a Ceres AD evaluator). Residual-only
// calls never touch the derivative source.
class delegating_residual_evaluator final : public residual_evaluator
{
public:
    delegating_residual_evaluator(residual_function residuals,
        std::unique_ptr<residual_evaluator>         derivative_source,
        api::derivative_mode                        source);

    const provider_metadata&   metadata() const override { return metadata_; }
    std::optional<std::string> last_error() const override;

protected:
    evaluation_status do_evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians) override;
    evaluation_status do_jacobian(
        const vector_type& x, const vector_type& residuals_at_x, matrix_type& jacobian) override;

private:
    residual_function                   residuals_;
    std::unique_ptr<residual_evaluator> inner_;
    provider_metadata                   metadata_;
    std::optional<std::string>          last_error_;
};

// Adapts a JacobianProvider (compute() yields residuals and Jacobian together).
std::unique_ptr<residual_evaluator> make_provider_evaluator(
    std::shared_ptr<JacobianProvider> provider);

// -- gradient evaluators ---------------------------------------------------------
class callback_gradient_evaluator final : public gradient_evaluator
{
public:
    callback_gradient_evaluator(std::size_t n,
        objective_function                  objective,
        gradient_function                   gradient,
        api::derivative_mode                source);

    std::size_t                num_parameters() const override { return n_; }
    api::derivative_mode       source() const override { return source_; }
    std::optional<std::string> last_error() const override { return last_error_; }

protected:
    evaluation_status do_evaluate(
        const vector_type& x, double& value, vector_type* gradient) override;
    evaluation_status do_gradient(const vector_type& x, vector_type& gradient) override;

private:
    std::size_t                n_;
    objective_function         objective_;
    gradient_function          gradient_;
    api::derivative_mode       source_;
    std::optional<std::string> last_error_;
};

class finite_difference_gradient_evaluator final : public gradient_evaluator
{
public:
    finite_difference_gradient_evaluator(std::size_t n,
        objective_function                           objective,
        double                                       relative_step = default_fd_step,
        api::bounds                                  bounds        = {});

    std::size_t          num_parameters() const override { return n_; }
    api::derivative_mode source() const override { return api::derivative_mode::finite_difference; }
    std::optional<std::string> last_error() const override { return last_error_; }

protected:
    evaluation_status do_evaluate(
        const vector_type& x, double& value, vector_type* gradient) override;
    evaluation_status do_gradient(const vector_type& x, vector_type& gradient) override;

private:
    std::size_t                n_;
    objective_function         objective_;
    double                     relative_step_;
    api::bounds                bounds_;
    std::optional<std::string> last_error_;
};

// -- the one derivative resolver --------------------------------------------------
// What every backend gets back from policy resolution. `evaluator` is the
// single-threaded per-solve object native kernels consume. `jacobian_callback`
// is a stateless-or-thread-safe callable for external backends that want a
// plain function (Ceres, TAO); it is empty when the backend should differentiate
// numerically itself or no source exists. `ad_factory` is set when the source is
// Ceres-native AD, so the Ceres backend can use its own cost function.
struct resolved_derivatives
{
    api::derivative_mode                    source = api::derivative_mode::automatic;
    std::unique_ptr<residual_evaluator>     evaluator;
    jacobian_function                       jacobian_callback;
    std::shared_ptr<const provider_factory> ad_factory;
};

struct resolved_gradient
{
    api::derivative_mode                source = api::derivative_mode::automatic;
    std::unique_ptr<gradient_evaluator> evaluator;
    // Bound to `evaluator` (so counters stay truthful); valid while this object lives.
    objective_function objective;
    gradient_function  gradient;
};

// Errors come back as the finished solver_result to return, never as raw
// pointers or exceptions.
std::variant<resolved_derivatives, solver_result> resolve_derivatives(
    const least_squares_problem& problem, const solve_options& options, const vector_type& x);

std::variant<resolved_gradient, solver_result> resolve_gradient(
    const optimization_problem& problem, const solve_options& options, const vector_type& x);
}  // namespace solverslib::api::detail

#endif  // SOLVERS_EVALUATORS_H_
