#pragma once

#include <cstddef>
#include <limits>

#include "include/detail/support.h"

namespace solverslib
{
/**
 * @brief Unified options for every scalar root-finding algorithm in
 * root_finding_algorithms, replacing the previous per-call trailing
 * parameter lists (tolerance_function, tolerance_parametes, max_iterations,
 * f_0) that dekker() and brent() each spelled out individually.
 *
 * function_offset lets a caller solve func(x) == function_offset directly
 * instead of folding the offset into the lambda passed to the solver.
 */
class SOLVER_VISIBILITY root_finding_options
{
    friend class root_finding_options_builder;

public:
    explicit root_finding_options(size_t max_iterations = 50,
        double                          tolerance_function  = std::numeric_limits<double>::epsilon(),
        double                          tolerance_parameter = std::numeric_limits<double>::epsilon(),
        double                          function_offset     = 0.0)
        : max_iterations_(max_iterations), tolerance_function_(tolerance_function),
          tolerance_parameter_(tolerance_parameter), function_offset_(function_offset)
    {
    }

    // Getters
    size_t max_iterations() const { return max_iterations_; }
    double tolerance_function() const { return tolerance_function_; }
    double tolerance_parameter() const { return tolerance_parameter_; }
    double function_offset() const { return function_offset_; }

    // Setters
    void set_max_iterations(size_t value) { max_iterations_ = value; }
    void set_tolerance_function(double value) { tolerance_function_ = value; }
    void set_tolerance_parameter(double value) { tolerance_parameter_ = value; }
    void set_function_offset(double value) { function_offset_ = value; }

private:
    size_t max_iterations_;
    double tolerance_function_;
    double tolerance_parameter_;
    double function_offset_;
};

/**
 * @brief Builder for root_finding_options configuration
 *
 * Usage:
 *   auto options = root_finding_options_builder()
 *       .with_max_iterations(100).with_tolerance_function(1e-12).build();
 */
class SOLVER_VISIBILITY root_finding_options_builder
{
public:
    root_finding_options_builder() = default;

    root_finding_options_builder& with_max_iterations(size_t value)
    {
        options_.set_max_iterations(value);
        return *this;
    }

    root_finding_options_builder& with_tolerance_function(double value)
    {
        options_.set_tolerance_function(value);
        return *this;
    }

    root_finding_options_builder& with_tolerance_parameter(double value)
    {
        options_.set_tolerance_parameter(value);
        return *this;
    }

    root_finding_options_builder& with_function_offset(double value)
    {
        options_.set_function_offset(value);
        return *this;
    }

    root_finding_options build() const { return options_; }

private:
    root_finding_options options_;
};

}  // namespace solverslib
