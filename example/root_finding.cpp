#include <cstdio>

#include "solver_options/root_finding_options.h"
#include "solvers/root_finding_algorithms.h"

int main()
{
    const auto options = solverslib::root_finding_options_builder()
                             .with_tolerance_function(1e-12)
                             .with_tolerance_parameter(1e-12)
                             .with_max_iterations(100)
                             .build();

    auto f = [](double x) { return x * x - 2.0; };

    const auto r = solverslib::detail::run_brent(f, 0.0, 2.0, options);

    std::printf("brent: root = %.15f, converged = %s, iterations = %zu\n",
        r.root,
        r.outcome == solverslib::detail::root_outcome::converged ? "true" : "false",
        r.iterations);

    return r.outcome == solverslib::detail::root_outcome::converged ? 0 : 1;
}
