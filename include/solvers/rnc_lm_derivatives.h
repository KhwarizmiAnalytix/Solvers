#pragma once

#include "detail/eigen_support.h"
#include <functional>
#include <vector>

namespace solverslib
{
// Ordinary derivatives (not Taylor coefficients) at t=0 along
// x(t) = base + sum_{q=1}^{coefficients.size()} t^q/q! * coefficients[q-1].
// R[k] = d^k r(x(t))/dt^k; J[k] = d^k J(x(t))/dt^k.
// The curve coefficients are held constant when differentiating the base.
struct rnc_curve_derivatives
{
    std::vector<vector_type> residual;
    std::vector<matrix_type> jacobian;
};

// Requested order is 1..4. Supply R[0..order] and J[0..max(0,order-2)].
// Order 1 with an empty curve obtains the residual and Jacobian at base.
// Can be implemented analytically or with Taylor AD (rnc_autodiff.h).
using rnc_derivative_function = std::function<void(const vector_type& base,
    const std::vector<vector_type>&                                   coefficients,
    int                                                               order,
    rnc_curve_derivatives&                                            out)>;
}  // namespace solverslib
