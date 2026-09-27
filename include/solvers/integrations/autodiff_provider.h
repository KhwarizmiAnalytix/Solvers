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
#include <stdexcept>
#include <type_traits>

#include "detail/eigen_support.h"
#include "detail/support.h"
#include "solvers/api/derivative_provider.h"
#include "solvers/api/problem.h"
#include "solvers/integrations/ceres_autodiff.h"

namespace solverslib
{

#if defined(SOLVERS_HAS_CERES)

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
template <class Functor>
class AutoDiffJacobianProvider final : public api::JacobianProvider
{
public:
    AutoDiffJacobianProvider(const Functor& functor, std::size_t n, std::size_t m)
        : functor_(functor),
          n_(n),
          m_(m),
          factory_(
              std::make_shared<detail::CeresAutoDiffFactory<Functor>>(functor, n, m))
    {
    }

    void compute(
        const vector_type& x,
        vector_type&       residuals,
        matrix_type&       jacobian) const override
    {
        auto evaluator = factory_->create_evaluator();
        auto status    = evaluator->evaluate(x, residuals, &jacobian);
        if (status == api::detail::evaluation_status::fatal_error)
        {
            auto err = evaluator->last_error();
            throw std::runtime_error(
                err.value_or("AutoDiffJacobianProvider: fatal error in evaluation"));
        }
    }

    void residuals_only(const vector_type& x, vector_type& residuals) const override
    {
        auto evaluator = factory_->create_evaluator();
        auto status    = evaluator->evaluate(x, residuals, nullptr);
        if (status == api::detail::evaluation_status::fatal_error)
        {
            auto err = evaluator->last_error();
            throw std::runtime_error(
                err.value_or("AutoDiffJacobianProvider: fatal error in residual-only evaluation"));
        }
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
    Functor functor_;
    std::size_t n_, m_;
    std::shared_ptr<detail::CeresAutoDiffFactory<Functor>> factory_;
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
inline api::auto_diff_tag auto_diff()
{
    return {};
}

// Convenience factory: create a least_squares_problem from a templated functor,
// with the model stored for use with the no-arg auto_diff() sentinel.
//
// Usage:
//   auto problem = least_squares(MyModel{}, n_params, n_residuals);
//   problem.derivatives(api::auto_diff());   // instantiates AutoDiffJacobianProvider
//   auto result = api::solve(problem, x0);
template <class Functor>
api::least_squares_problem least_squares(
    const Functor& functor, std::size_t n, std::size_t m)
{
    api::least_squares_problem p;
    p.num_parameters = n;
    p.num_residuals  = m;

    // Double-precision residuals callback (no AD types required)
    p.residuals = [functor](const vector_type& x, vector_type& r) {
        if (!functor(x.data(), r.data()))
        {
            throw std::runtime_error("least_squares model functor returned false");
        }
    };

    // Store a factory so problem.derivatives(auto_diff()) can instantiate
    // the AutoDiffJacobianProvider later without knowing the Functor type.
    p.model_provider_factory =
        [functor, n, m]() -> std::shared_ptr<api::JacobianProvider> {
        return std::make_shared<AutoDiffJacobianProvider<Functor>>(functor, n, m);
    };

    return p;
}

#else  // !SOLVERS_HAS_CERES

// Stubs when Ceres is not available.

template <class Functor>
class AutoDiffJacobianProvider final : public api::JacobianProvider
{
public:
    AutoDiffJacobianProvider(const Functor&, std::size_t, std::size_t)
    {
        static_assert(
            sizeof(Functor) == 0,
            "AutoDiffJacobianProvider requires SOLVERS_ENABLE_CERES=ON");
    }

    void        compute(const vector_type&, vector_type&, matrix_type&) const override {}
    std::size_t num_parameters() const override { return 0; }
    std::size_t num_residuals() const override { return 0; }
    api::derivative_mode source() const override
    {
        return api::derivative_mode::automatic_differentiation;
    }
};

template <class Functor>
std::shared_ptr<AutoDiffJacobianProvider<Functor>> auto_diff(
    const Functor&, std::size_t, std::size_t)
{
    static_assert(
        sizeof(Functor) == 0,
        "auto_diff() requires SOLVERS_ENABLE_CERES=ON");
    return nullptr;
}

inline api::auto_diff_tag auto_diff()
{
    return {};
}

template <class Functor>
api::least_squares_problem least_squares(const Functor&, std::size_t, std::size_t)
{
    static_assert(
        sizeof(Functor) == 0,
        "least_squares() convenience factory requires SOLVERS_ENABLE_CERES=ON. "
        "Without Ceres, create least_squares_problem directly and supply an "
        "analytic Jacobian or finite_difference() provider.");
    return {};
}

#endif  // SOLVERS_HAS_CERES

}  // namespace solverslib

#endif  // SOLVERS_AUTODIFF_PROVIDER_H_
