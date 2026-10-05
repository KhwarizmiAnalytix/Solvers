#ifndef SOLVERS_AUTODIFF_PROVIDER_H_
#define SOLVERS_AUTODIFF_PROVIDER_H_

// AutoDiff Jacobian provider — wraps Ceres' DynamicAutoDiffCostFunction to
// implement the JacobianProvider interface. The Ceres dependency is hidden
// behind this header; the rest of the solver API never includes Ceres types.
//
// Requires SOLVERS_HAS_CERES. Include this header only in translation units
// that need to construct an autodiff provider; it must not appear in generic
// solver headers.

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "api/derivative_provider.h"
#include "api/problem.h"
#include "solvers/integrations/ceres_autodiff.h"

namespace solverslib
{

#if SOLVERS_HAS_CERES

// AutoDiffJacobianProvider<Functor>
//
// Wraps a Ceres-style templated functor (operator()(const T* x, T* r)) as a
// JacobianProvider. Internally uses CeresAutoDiffFactory/Evaluator so the
// Ceres backend can take the native autodiff path while the native backends
// (LM, GN, L-BFGS) consume the provider through the standard interface.
//
// Functor contract:
//   template <typename T>
//   bool operator()(const T* const x, T* residuals) const;
//
template <class Functor> class AutoDiffJacobianProvider final : public api::JacobianProvider
{
public:
    AutoDiffJacobianProvider(const Functor& functor, std::size_t n, std::size_t m)
        : functor_(functor), n_(n), m_(m),
          factory_(std::make_shared<detail::CeresAutoDiffFactory<Functor>>(functor, n, m))
    {
    }

    // One Ceres cost function is built lazily and reused; evaluation is
    // serialized by a mutex, so a provider shared between concurrent solves is
    // safe but not parallel. For parallel use, give each solve its own evaluator
    // through ceres_factory()->create_evaluator().
    void compute(const vector_type& x, vector_type& residuals, matrix_type& jacobian) const override
    {
        evaluate_locked(x, residuals, &jacobian, "fatal error in evaluation");
    }

    void residuals_only(const vector_type& x, vector_type& residuals) const override
    {
        evaluate_locked(x, residuals, nullptr, "fatal error in residual-only evaluation");
    }

    std::size_t num_parameters() const override { return n_; }
    std::size_t num_residuals() const override { return m_; }

    api::derivative_mode source() const override
    {
        return api::derivative_mode::automatic_differentiation;
    }

    // Expose the underlying Ceres factory so the Ceres backend can use its
    // native autodiff path rather than routing through this interface.
    std::shared_ptr<const api::detail::provider_factory> ceres_factory() const override
    {
        return factory_;
    }

private:
    void evaluate_locked(
        const vector_type& x, vector_type& residuals, matrix_type* jacobian, const char* what) const
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!evaluator_)
        {
            evaluator_ = factory_->create_evaluator();
        }
        const auto status = evaluator_->evaluate(x, residuals, jacobian);
        if (status == api::detail::evaluation_status::fatal_error)
        {
            throw std::runtime_error(evaluator_->last_error().value_or(
                std::string("AutoDiffJacobianProvider: ") + what));
        }
        if (status == api::detail::evaluation_status::invalid_trial)
        {
            throw std::runtime_error(
                "AutoDiffJacobianProvider: functor returned false (invalid trial point)");
        }
    }

    Functor                                                  functor_;
    std::size_t                                              n_, m_;
    std::shared_ptr<detail::CeresAutoDiffFactory<Functor>>   factory_;
    mutable std::mutex                                       mutex_;
    mutable std::unique_ptr<api::detail::residual_evaluator> evaluator_;
};

// Factory function: create an AutoDiffJacobianProvider from a templated functor.
// This is the recommended explicit form when not using the least_squares() helper.
//
// Usage:
//   problem.set_jacobian_provider(auto_diff(MyModel{}, n_params, n_residuals));
template <class Functor>
std::shared_ptr<AutoDiffJacobianProvider<Functor>> auto_diff(
    const Functor& functor, std::size_t n, std::size_t m)
{
    return std::make_shared<AutoDiffJacobianProvider<Functor>>(functor, n, m);
}

// Zero-arg sentinel for use with least_squares(model, n, m):
//   auto problem = least_squares(MyModel{}, n, m);
//   problem.derivatives(auto_diff());
// The sentinel itself lives in solverslib::api; this re-exports it so both the
// 3-argument factory above and auto_diff() resolve in solverslib.
using api::auto_diff;

// Convenience factory: create a least_squares_problem from a templated functor,
// with the model stored for use with the no-arg auto_diff() sentinel.
//
// Usage:
//   auto problem = least_squares(MyModel{}, n_params, n_residuals);
//   problem.derivatives(api::auto_diff());   // instantiates AutoDiffJacobianProvider
//   auto result = api::solve(problem, x0);
template <class Functor>
api::least_squares_problem least_squares(const Functor& functor, std::size_t n, std::size_t m)
{
    api::least_squares_problem p;
    p.num_parameters = n;
    p.num_residuals  = m;

    // Double-precision residuals callback (no AD types required)
    // Model failures intentionally propagate to the solver boundary.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    p.residuals = [functor](const vector_type& x, vector_type& r)
    {
        if (!functor(x.data(), r.data()))
        {
            throw std::runtime_error("least_squares model functor returned false");
        }
    };

    // Store a factory so problem.derivatives(auto_diff()) can instantiate
    // the AutoDiffJacobianProvider later without knowing the Functor type.
    // Allocation failures intentionally propagate to the caller.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    p.set_model_provider_factory([functor, n, m]() -> std::shared_ptr<api::JacobianProvider>
        { return std::make_shared<AutoDiffJacobianProvider<Functor>>(functor, n, m); });

    return p;
}

#else  // !SOLVERS_HAS_CERES

// Stubs when Ceres is not available.

template <class Functor> class AutoDiffJacobianProvider final : public api::JacobianProvider
{
public:
    AutoDiffJacobianProvider(const Functor&, std::size_t, std::size_t)
    {
        static_assert(
            sizeof(Functor) == 0, "AutoDiffJacobianProvider requires SOLVERS_ENABLE_CERES=ON");
    }

    void                 compute(const vector_type&, vector_type&, matrix_type&) const override {}
    std::size_t          num_parameters() const override { return 0; }
    std::size_t          num_residuals() const override { return 0; }
    api::derivative_mode source() const override
    {
        return api::derivative_mode::automatic_differentiation;
    }
};

template <class Functor>
std::shared_ptr<AutoDiffJacobianProvider<Functor>> auto_diff(
    const Functor&, std::size_t, std::size_t)
{
    static_assert(sizeof(Functor) == 0, "auto_diff() requires SOLVERS_ENABLE_CERES=ON");
    return nullptr;
}

using api::auto_diff;

template <class Functor>
api::least_squares_problem least_squares(const Functor&, std::size_t, std::size_t)
{
    static_assert(sizeof(Functor) == 0,
        "least_squares() convenience factory requires SOLVERS_ENABLE_CERES=ON. "
        "Without Ceres, create least_squares_problem directly and supply an "
        "analytic Jacobian or finite_difference() provider.");
    return {};
}

#endif  // SOLVERS_HAS_CERES

}  // namespace solverslib

#endif  // SOLVERS_AUTODIFF_PROVIDER_H_
