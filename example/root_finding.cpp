#include <cstdio>

#include "solvers/root_finding_algorithms.h"

int main()
{
    double root = 0.0;

    auto options = solverslib::root_finding_options_builder()
                       .with_tolerance_function(1e-12)
                       .with_tolerance_parameter(1e-12)
                       .with_max_iterations(100)
                       .build();

    auto f = [](double x) { return x * x - 2.0; };

    bool converged =
        solverslib::root_finding_algorithms::brent(f, 0.0, 2.0, root, options);

    std::printf("brent: root = %.15f, converged = %s\n",
        root, converged ? "true" : "false");

    return converged ? 0 : 1;
}
