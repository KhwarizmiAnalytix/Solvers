#include "solvers/api/solve.h"
#include "solvers/integrations/rnc_autodiff.h"
#include <cmath>
#include <iostream>

struct Valley
{
    int                     power;
    template <class T> bool operator()(const T* x, T* r) const
    {
        using std::pow;
        r[0] = x[0];
        r[1] = T(1e6) * (x[1] - pow(x[0], power) / T(power));
        return true;
    }
};

int main()
{
    using namespace solverslib;
    std::cout << "power,order,iterations,accepted,rejected,cost,status\n";
    for (int power : {2, 3})
        for (int order : {1, 2, 3, 4})
        {
            auto               problem = rnc_least_squares(Valley{power}, 2, 2);
            api::solve_options options;
            options.algorithm      = api::algorithm::riemann_normal_coordinate_lm;
            options.rnc_lm         = api::rnc_lm_options{};
            options.rnc_lm->order  = order;
            options.max_iterations = 20000;
            // Match the objective threshold of Eq. (6)'s experiment: C < 1e-4.
            options.function_tolerance = std::sqrt(2e-4);
            vector_type start(2);
            start << 1., 1. / power;
            const auto result = api::solve(problem, start, options);
            std::cout << power << ',' << order << ',' << result.iterations << ','
                      << result.accepted_steps.value_or(0) << ','
                      << result.rejected_steps.value_or(0) << ',' << result.objective << ','
                      << api::to_string(result.status) << '\n';
            if (!result.converged())
                return 1;
        }
}
