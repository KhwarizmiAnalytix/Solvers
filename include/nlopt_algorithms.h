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
class nlopt_algorithms
{
public:
    using ObjFunc = std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using ConFunc = std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>;

    MATH_API nlopt_algorithms(
        ObjFunc func, ConFunc func_con, std::vector<double> lb, std::vector<double> ub);

    MATH_API bool is_supported();

    MATH_API void solve(std::vector<double>& parameters, const solver_options_nlopt& options);

private:
    std::vector<double> bl_;
    std::vector<double> bu_;
    ObjFunc             objfun_;
    ConFunc             confun_;

    static double OBJFUN(const std::vector<double>& x, std::vector<double>& grad, void* data);

    static double CONFUN(const std::vector<double>& x, std::vector<double>& grad, void* data);
};
}  // namespace solverslib
