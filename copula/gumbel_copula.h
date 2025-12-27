#pragma once

#include <cmath>
#include <vector>

#include "common/wrapping_hints.h"
#include "copula/copula.h"

namespace quarisma
{
class MATH_VISIBILITY gumbel_copula : public copula
{
public:
    MATH_API explicit gumbel_copula(double theta);

    ~gumbel_copula() override = default;

    MATH_API double evaluate(const std::vector<double>& u) const override;

    MATH_API double density(const std::vector<double>& u) const override;

    MATH_API void random_sample(
        double*                output,
        size_t                 n,
        uniforms_functoin_type uniforms_func,
        size_t                 skip_count = 0) const override;

    MATH_API size_t dimension() const override;

    MATH_API double theta() const;

private:
    double theta_;
};

}  // namespace quarisma