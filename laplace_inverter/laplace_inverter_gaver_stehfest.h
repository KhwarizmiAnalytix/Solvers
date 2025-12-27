#pragma once

#include <cstddef>
#include <vector>

#include "MathModule.h"

namespace quarisma
{
class laplace_inverter_gaver_stehfest
{
public:
    MATH_API laplace_inverter_gaver_stehfest(size_t size, double shift);  // constructor

    MATH_API double operator()(double (*F)(double, double), double t, double r) const;

private:
    double              shift_;
    std::vector<double> coefficient_;
};
}  // namespace quarisma
