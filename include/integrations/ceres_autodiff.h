#ifndef SOLVERS_CERES_AUTODIFF_H_
#define SOLVERS_CERES_AUTODIFF_H_

#if SOLVERS_HAS_CERES

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "detail/eigen_support.h"
#include "api/detail/evaluator.h"

namespace solverslib
{

// One-block functor adapter for Ceres DynamicAutoDiffCostFunction.
// Maps the single-array interface (const T* x, T* residuals) to Ceres'
// multi-block interface (const T* const* blocks, T* residuals).
template <class Functor> struct OneBlockFunctor
{
    Functor functor;

    template <class T> bool operator()(T const* const* blocks, T* residuals) const
    {
        return functor(blocks[0], residuals);
    }
};

namespace detail
{

// Concrete evaluator using Ceres' DynamicAutoDiffCostFunction.
// Instantiated at compile time (caller knows the functor type).
template <class Functor> class CeresAutoDiffEvaluator : public api::detail::residual_evaluator
{
public:
    explicit CeresAutoDiffEvaluator(const Functor& f, std::size_t n, std::size_t m)
        : functor_(f), metadata_{n, m, api::derivative_mode::automatic_differentiation, true}
    {
        // Construct Ceres AD cost function with stride 4
        // (ceil(n/4) Jet passes for n parameters)
        using Adapter = OneBlockFunctor<Functor>;
        cost_function_ =
            std::make_unique<ceres::DynamicAutoDiffCostFunction<Adapter, 4>>(new Adapter{functor_});
        cost_function_->AddParameterBlock(static_cast<int>(n));
        cost_function_->SetNumResiduals(static_cast<int>(m));
    }

    const api::detail::provider_metadata& metadata() const override { return metadata_; }

    std::optional<std::string> last_error() const override { return last_error_; }

protected:
    api::detail::evaluation_status do_evaluate(
        const vector_type& x, vector_type& residuals, matrix_type* jacobians) override
    {
        try
        {
            resize_if_needed(residuals, metadata_.num_residuals);

            // Vectors are contiguous, so Ceres reads and writes them in place.
            double const* params[]        = {x.data()};
            double*       jacobians_arr[] = {nullptr};
            if (jacobians != nullptr)
            {
                // Ceres writes row-major; the scratch buffer is reused across calls.
                jacobian_scratch_.resize(metadata_.num_residuals * metadata_.num_parameters);
                jacobians_arr[0] = jacobian_scratch_.data();
            }

            if (!cost_function_->Evaluate(params, residuals.data(), jacobians_arr))
            {
                return api::detail::evaluation_status::invalid_trial;
            }
            if (!residuals.allFinite())
            {
                last_error_ = "Non-finite residual computed";
                return api::detail::evaluation_status::fatal_error;
            }

            if (jacobians != nullptr)
            {
                copy_from_row_major(*jacobians,
                    jacobian_scratch_.data(),
                    metadata_.num_residuals,
                    metadata_.num_parameters);
                if (!jacobians->allFinite())
                {
                    last_error_ = "Non-finite Jacobian entry computed";
                    return api::detail::evaluation_status::fatal_error;
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

private:
    Functor                                                                          functor_;
    std::unique_ptr<ceres::DynamicAutoDiffCostFunction<OneBlockFunctor<Functor>, 4>> cost_function_;
    api::detail::provider_metadata                                                   metadata_;
    std::vector<double>                jacobian_scratch_;
    mutable std::optional<std::string> last_error_;
};

// Factory for creating Ceres AD evaluators.
// Stores the functor and dimensions; creates evaluators on demand.
template <class Functor> class CeresAutoDiffFactory : public api::detail::provider_factory
{
public:
    CeresAutoDiffFactory(const Functor& f, std::size_t n, std::size_t m)
        : functor_(f), n_(n), m_(m),
          metadata_{n, m, api::derivative_mode::automatic_differentiation, true}
    {
    }

    const api::detail::provider_metadata& metadata() const override { return metadata_; }

    std::unique_ptr<api::detail::residual_evaluator> create_evaluator() const override
    {
        return std::make_unique<CeresAutoDiffEvaluator<Functor>>(functor_, n_, m_);
    }

private:
    Functor                        functor_;
    std::size_t                    n_, m_;
    api::detail::provider_metadata metadata_;
};

}  // namespace detail

}  // namespace solverslib

#endif  // SOLVERS_HAS_CERES

#endif  // SOLVERS_CERES_AUTODIFF_H_
