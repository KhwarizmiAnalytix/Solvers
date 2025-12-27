#include "common/discretization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "smp/tools.h"

namespace quarisma::discretization
{

//-----------------------------------------------------------------------------
void uniform(
    double* output, size_t n, double value_min, double value_max, double value, int j0, bool center)
{
    if (n == 0)
    {
        return;
    }

    if (n == 1)
    {
        output[0] = value_min;
        output[1] = value_max;
        return;
    }

    if (j0 == -1)
    {
        const size_t half_size = n / 2;
        const double dx1       = 2.0 * (value - value_min) / static_cast<double>(n - 1);
        const double dx2       = 2.0 * (value_max - value) / static_cast<double>(n - 1);

        for (size_t i = 0; i < half_size; ++i)
        {
            output[i] = value_min + static_cast<double>(i) * dx1;
        }

        output[half_size] = value;

        for (size_t i = half_size + 1; i < n; ++i)
        {
            output[i] = value + static_cast<double>(i - half_size) * dx2;
        }
        return;
    }

    if (center)
    {
        auto h = (value - value_min) / (static_cast<double>(j0) - 0.5);

        value_max = static_cast<double>(n - 2) * h + value_min;

        output[0] = value_min;
        for (int i = 1; i < n - 1; ++i)
        {
            output[i] = value_min + (i - 0.5) * h;
        }
        output[n - 1] = value_max;
    }
    else
    {
        const double dx1 = (value - value_min) / static_cast<double>(j0);
        const double dx2 = (value_max - value) / static_cast<double>(n - j0);

        for (size_t i = 0; i < j0; ++i)
        {
            output[i] = value_min + static_cast<double>(i) * dx1;
        }

        output[j0] = value;

        for (size_t i = j0 + 1; i < n; ++i)
        {
            output[i] = value + static_cast<double>(i - j0) * dx2;
        }
    }
}

//-----------------------------------------------------------------------------
void asinh(
    double* output, size_t n, double value_min, double value_max, double value, double scaling)
{
    const double range = value_max - value_min;
    const double a_min = std::asinh((value_min - value) / (range * scaling));
    const double a_max = std::asinh((value_max - value) / (range * scaling));

    uniform(output, n, a_min, a_max, 0.0);

    for (size_t i = 0; i < n; ++i)
    {
        output[i] = value + scaling * std::sinh(output[i]);
    }
}

//-----------------------------------------------------------------------------
void extend_vector_uniform(double* output, const double* input, size_t n, size_t m)
{
    if (n >= m)
    {
        throw std::invalid_argument("m must be greater than the input vector size");
    }

    std::vector<double> result;
    result.reserve(m);

    const size_t total_new_elements = m - n;
    const size_t base_insert_count  = total_new_elements / (n - 1);
    const size_t remainder          = total_new_elements % (n - 1);

    double start = input[0];

    size_t idx = 0;
    for (size_t i = 0; i < n - 1; ++i)
    {
        const size_t insert_count = base_insert_count + (i < remainder ? 1 : 0);

        const double end  = input[i + 1];
        const double step = (end - start) / static_cast<double>(insert_count + 1);

        output[idx++] = start;
        for (size_t j = 1; j <= insert_count; ++j)
        {
            output[idx++] = (start + static_cast<double>(j) * step);
        }

        start = end;
    }

    output[idx] = input[n - 1];
}

}  // namespace quarisma::discretization