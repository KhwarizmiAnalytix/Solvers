#ifndef SOLVERS_CERES_AUTODIFF_H_
#define SOLVERS_CERES_AUTODIFF_H_

#if defined(SOLVERS_HAS_CERES)

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "detail/eigen_support.h"
#include "solvers/api/detail/evaluator.h"
#include "solvers/api/problem.h"

namespace solverslib
{

// One-block functor adapter for Ceres DynamicAutoDiffCostFunction.
// Maps the single-array interface (const T* x, T* residuals) to Ceres'
// multi-block interface (const T* const* blocks, T* residuals).
template <class Functor>
struct OneBlockFunctor
{
    Functor functor;

    template <class T>
    bool operator()(T const* const* blocks, T* residuals) const
    {
        return functor(blocks[0], residuals);
    }
};

namespace detail
{

// Concrete evaluator using Ceres' DynamicAutoDiffCostFunction.
// Instantiated at compile time (caller knows the functor type).
template <class Functor>
class CeresAutoDiffEvaluator : public api::detail::residual_evaluator
{
public:
    explicit CeresAutoDiffEvaluator(const Functor& f, std::size_t n, std::size_t m)
        : functor_(f), metadata_{n, m, api::derivative_mode::automatic_differentiation, true}
    {
        // Construct Ceres AD cost function with stride 4
        // (ceil(n/4) Jet passes for n parameters)
        using Adapter = OneBlockFunctor<Functor>;
        cost_function_ = std::make_unique<ceres::DynamicAutoDiffCostFunction<Adapter, 4>>(
            new Adapter{functor_});
        cost_function_->AddParameterBlock(static_cast<int>(n));
        cost_function_->SetNumResiduals(static_cast<int>(m));
    }

    api::detail::evaluation_status evaluate(
        const vector_type& x,
        vector_type& residuals,
        matrix_type* jacobians = nullptr) override
    {
        try
        {
            // Convert Eigen vectors to raw pointers for Ceres
            std::vector<double> x_vec(x.data(), x.data() + x.size());
            std::vector<double> r_vec(residuals.size());

            double const* params[] = {x_vec.data()};
            double* jacobians_arr[] = {jacobians ? jacobians->data() : nullptr};

            // Evaluate through Ceres interface
            bool ok = cost_function_->Evaluate(params, r_vec.data(), jacobians_arr);
            if (!ok)
            {
                return api::detail::evaluation_status::invalid_trial;
            }

            // Copy residuals back and validate finiteness
            for (std::size_t i = 0; i < r_vec.size(); ++i)
            {
                if (!std::isfinite(r_vec[i]))
                {
                    last_error_ = "Non-finite residual computed";
                    return api::detail::evaluation_status::fatal_error;
                }
                residuals[i] = r_vec[i];
            }

            // Validate Jacobian if requested
            if (jacobians)
            {
                for (std::size_t i = 0; i < jacobians->size(); ++i)
                {
                    if (!std::isfinite((*jacobians)(i / jacobians->cols(), i % jacobians->cols())))
                    {
                        last_error_ = "Non-finite Jacobian entry computed";
                        return api::detail::evaluation_status::fatal_error;
                    }
                }
            }

            return api::detail::evaluation_status::ok;
        }
        catch (const std::exception& e)
        {
            last_error_ = std::string("Ceres AD exception: ") + e.what();
            return api::detail::evaluation_status::fatal_error;
        }
        catch (...)
        {
            last_error_ = "Ceres AD unknown exception";
            return api::detail::evaluation_status::fatal_error;
        }
    }

    const api::detail::provider_metadata& metadata() const override { return metadata_; }

    std::optional<std::string> last_error() const override { return last_error_; }

private:
    Functor functor_;
    std::unique_ptr<ceres::DynamicAutoDiffCostFunction<OneBlockFunctor<Functor>, 4>>
        cost_function_;
    api::detail::provider_metadata metadata_;
    mutable std::optional<std::string> last_error_;
};

// Factory for creating Ceres AD evaluators.
// Stores the functor and dimensions; creates evaluators on demand.
template <class Functor>
class CeresAutoDiffFactory : public api::detail::provider_factory
{
public:
    CeresAutoDiffFactory(const Functor& f, std::size_t n, std::size_t m)
        : functor_(f), n_(n), m_(m),
          metadata_{n, m, api::derivative_mode::automatic_differentiation, true}
    {
    }

    const api::detail::provider_metadata& metadata() const override { return metadata_; }

    std::unique_ptr<api::detail::residual_evaluator> create_evaluator() override
    {
        return std::make_unique<CeresAutoDiffEvaluator<Functor>>(functor_, n_, m_);
    }

private:
    Functor functor_;
    std::size_t n_, m_;
    api::detail::provider_metadata metadata_;
};

}  // namespace detail

// Public API: Create a least-squares problem with Ceres AD provider.
// The templated functor must have:
//   template <typename T>
//   bool operator()(const T* const x, T* residuals) const
// It will be instantiated with both double (residual computation) and
// Ceres Jet types (automatic differentiation).
template <class Functor>
api::least_squares_problem make_ceres_autodiff_problem(
    std::size_t num_parameters,
    std::size_t num_residuals,
    const Functor& templated_residuals)
{
    api::least_squares_problem problem;
    problem.num_parameters = num_parameters;
    problem.num_residuals  = num_residuals;

    // Store the functor for potential legacy access
    problem.set_templated_residuals(templated_residuals);

    // Create and attach the provider factory
    // The solver will use this to instantiate the AD evaluator
    problem.provider_factory = std::make_shared<detail::CeresAutoDiffFactory<Functor>>(
        templated_residuals, num_parameters, num_residuals);

    return problem;
}

}  // namespace solverslib

#else  // !SOLVERS_HAS_CERES

// Stub when Ceres is not available
namespace solverslib
{
template <class Functor>
api::least_squares_problem make_ceres_autodiff_problem(
    std::size_t,
    std::size_t,
    const Functor&)
{
    static_assert(false,
        "Ceres backend not compiled in; using solvers/integrations/ceres_autodiff.h "
        "requires SOLVERS_ENABLE_CERES=ON");
}
}

#endif  // SOLVERS_HAS_CERES

#endif  // SOLVERS_CERES_AUTODIFF_H_
