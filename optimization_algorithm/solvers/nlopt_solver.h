#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "MathModule.h"
#include "common/macros.h"

namespace quarisma
{
template <typename value_t>
class vector;
template <typename value_t>
class matrix;
class solver_options_nlopt;
}  // namespace quarisma

namespace quarisma
{
class nlopt_solver
{
public:
    using ObjFunc     = std::function<void(const quarisma::vector<double>&, quarisma::vector<double>&)>;
    using ObjFunc_aad = std::function<void(const quarisma::vector<double>&, quarisma::matrix<double>&)>;
    //using ConFunc = std::function<double(const quarisma::vector<double>&, quarisma::vector<double>&)>;

    MATH_API nlopt_solver(
        size_t              num_parameters,
        size_t              num_residuals,
        ObjFunc             func,
        ObjFunc_aad         func_aad,
        std::vector<double> lb = {},
        std::vector<double> ub = {});

    MATH_API static bool is_supported();

    MATH_API void solve(std::vector<double>& parameters, const solver_options_nlopt& options);

private:
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    size_t              num_parameters_;
    size_t              num_residuals_;
    ObjFunc             objfun_;
    ObjFunc_aad         objfun_aad_;

    static double OBJFUN(const std::vector<double>& x, std::vector<double>& grad, void* data);

    static double CONFUN(const std::vector<double>& x, std::vector<double>& grad, void* data);
};
}  // namespace quarisma
