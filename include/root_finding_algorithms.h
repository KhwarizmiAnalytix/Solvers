#pragma once

#include <cstddef>
#include <functional>
#include <limits>

#include "include/detail/support.h"
#include "include/detail/support.h"

namespace solverslib
{
class root_finding_algorithms
{
public:
    using function_type          = std::function<double(double)>;
    using function_gradient_type = std::function<double(double, double&)>;

    MATH_API static bool dekker(
        function_gradient_type const& func,
        double                        x1,
        double                        x2,
        double&                       result,
        double                        tolerance_function  = std::numeric_limits<double>::epsilon(),
        double                        tolerance_parametes = std::numeric_limits<double>::epsilon(),
        size_t                        max_iterations      = 50);

    MATH_API static bool brent(
        function_type const& func,
        double               x1,
        double               x2,
        double&              root,
        double               f_0                 = 0.,
        double               tolerance_function  = std::numeric_limits<double>::epsilon(),
        double               tolerance_parametes = std::numeric_limits<double>::epsilon(),
        size_t               max_iterations      = 50);

private:
    SOLVERS_DELETE_CLASS(root_finding_algorithms);
};
};  // namespace solverslib
