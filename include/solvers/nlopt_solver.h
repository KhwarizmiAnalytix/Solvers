#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "include/detail/support.h"
#include "include/detail/support.h"

namespace solverslib
{


class solver_options_nlopt;
}  // namespace solverslib

namespace solverslib
{
class nlopt_solver
{
public:
    using ObjFunc     = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using ObjFunc_aad = std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)>;
    //using ConFunc = std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>;

    SOLVER_API nlopt_solver(
        size_t              num_parameters,
        size_t              num_residuals,
        ObjFunc             func,
        ObjFunc_aad         func_aad,
        std::vector<double> lb = {},
        std::vector<double> ub = {});

    SOLVER_API static bool is_supported();

    SOLVER_API void solve(std::vector<double>& parameters, const solver_options_nlopt& options);

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
}  // namespace solverslib
