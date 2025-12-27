#include "common/tridiagonal_operations.h"

#include "common/macros.h"
#include "smp/tools.h"

namespace quarisma::tridiagonal_operations
{
//-----------------------------------------------------------------------------
void decomposition(
    matrix<double>& output,
    const size_t    outer_dim,
    const size_t    dim,
    const size_t    inner_dim,
    const bool      parallelize)
{
    QUARISMA_CHECK_DEBUG(
        output.rows() == 3, "tridiagonal root_finding_algorithms dimension is not 3!");

    const auto n = dim * inner_dim;

    auto parallel_eval = [&output, inner_dim, n, outer_dim](int from, int to)
    {
        const auto offset = static_cast<size_t>(from) * n;

        auto* __restrict itr0 = output[0].data() + offset;
        auto* __restrict itr1 = output[1].data() + offset;
        auto* __restrict itr2 = output[2].data() + offset;

        for (size_t l = from; l < to; l++)
        {
            for (size_t i = 0; i < inner_dim; ++i)
            {
                const double inv_diag = 1. / (*itr1);

                *itr0++ = inv_diag;
                *itr1++ = 0.;
                *itr2++ *= inv_diag;
            }
            for (size_t i = inner_dim; i < n; ++i)
            {
                const auto tmp      = *(itr2 - inner_dim);
                const auto alpha    = -*itr0;
                const auto inv_diag = 1. / (*itr1 + alpha * tmp);

                *itr0++ = inv_diag;
                *itr1++ = alpha * inv_diag;
                *itr2++ *= inv_diag;
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }

    output[2] = -output[2];
};

//-----------------------------------------------------------------------------
void decomposition_aad(
    matrix<double>&       output_aad,
    const matrix<double>& output,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize)
{
    output_aad[2] = -output_aad[2];
    QUARISMA_CHECK_DEBUG(
        output.rows() == 3, "tridiagonal root_finding_algorithms dimension is not 3!");

    const auto n = dim * inner_dim;

    auto parallel_eval = [&output, &output_aad, inner_dim, n, outer_dim](int from, int to)
    {
        const auto offset = static_cast<size_t>(to) * n - 1;

        auto const* __restrict itr0 = output[0].data() + offset;
        auto const* __restrict itr1 = output[1].data() + offset;
        auto const* __restrict itr2 = output[2].data() + offset;

        auto* __restrict itr0_aad = output_aad[0].data() + offset;
        auto* __restrict itr1_aad = output_aad[1].data() + offset;
        auto* __restrict itr2_aad = output_aad[2].data() + offset;

        for (size_t l = to; l > from; --l)
        {
            for (size_t i = n; i > inner_dim;
                 --i, --itr0, --itr1, --itr2, --itr0_aad, --itr1_aad, --itr2_aad)
            {
                const auto inv_diag = *itr0;
                const auto alpha    = *itr1 / inv_diag;
                const auto tmp      = -*(itr2 - inner_dim);
                const auto tmp_itr2 = -*itr2 / inv_diag;

                //*itr_r2 = *itr2 * inv_diag;
                auto inv_diag_aad = *itr2_aad * tmp_itr2;
                *itr2_aad *= inv_diag;

                //*itr_r1 = alpha * inv_diag;
                inv_diag_aad += *itr1_aad * alpha;
                auto alpha_aad = *itr1_aad * inv_diag;

                //*itr_r0 = inv_diag;
                inv_diag_aad += *itr0_aad;

                inv_diag_aad *= -(inv_diag * inv_diag);
                //inv_diag = 1. / (*itr1 + alpha * tmp);
                *itr1_aad = inv_diag_aad;
                alpha_aad += inv_diag_aad * tmp;
                auto tmp_aad = inv_diag_aad * alpha;

                //alpha = -*itr0;
                *itr0_aad = -alpha_aad;

                //tmp = *(itr2 - inner_dim);
                *(itr2_aad - inner_dim) += tmp_aad;
            }
            for (size_t i = inner_dim; i > 0;
                 --i, --itr0, --itr1, --itr2, --itr0_aad, --itr1_aad, --itr2_aad)
            {
                const double inv_diag = *itr0;

                //*itr1++ = 0.;
                *itr1_aad = 0.;

                auto tmp_itr2 = -*itr2 / inv_diag;
                //*itr2++ *= inv_diag;
                auto inv_diag_aad = *itr2_aad * tmp_itr2;
                *itr2_aad         = *itr2_aad * inv_diag;

                //*itr0++ = inv_diag;
                inv_diag_aad += *itr0_aad;

                //const double inv_diag = 1. / (*itr1);
                *itr1_aad = -inv_diag_aad * inv_diag * inv_diag;
                *itr0_aad = 0.;
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
};

//-----------------------------------------------------------------------------
void solve_decomposed(
    vector<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize)
{
    const auto n   = inner_dim * dim;
    const int  end = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;

    x *= decomposed[0];

    auto parallel_eval = [&x, &decomposed, inner_dim, n, end](int from, int to)
    {
        auto* __restrict itr_x      = x.data();
        auto const* __restrict itr1 = decomposed[1].data();
        auto const* __restrict itr2 = decomposed[2].data();

        for (int l = from; l < to; ++l)
        {
            const auto offset               = static_cast<size_t>(l) * n;
            auto* __restrict tmp            = itr_x + offset;
            auto const* __restrict tmp_itr1 = itr1 + offset;
            auto const* __restrict tmp_itr2 = itr2 + offset;

            for (size_t i = inner_dim; i < n; ++i)
            {
                tmp[i] += tmp_itr1[i] * tmp[i - inner_dim];
            }

            for (int j = end; j >= 0; --j)
            {
                tmp[j] += tmp_itr2[j] * tmp[j + inner_dim];
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
}

//-----------------------------------------------------------------------------
void solve_decomposed_vectorised(
    matrix<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize)
{
    const auto n   = inner_dim * dim;
    const int  end = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;

    size_t rows    = x.rows();
    size_t columns = x.columns();

    for (size_t i = 0; i < rows; ++i)
    {
        x[i] = x[i] * (decomposed[0][i]);
    }

    auto parallel_eval = [&x, &decomposed, inner_dim, n, end, rows, columns](int from, int to)
    {
        auto* __restrict itr_x      = x.data();
        auto const* __restrict itr1 = decomposed[1].data();
        auto const* __restrict itr2 = decomposed[2].data();

        for (int l = from; l < to; ++l)
        {
            const auto offset               = static_cast<size_t>(l) * n;
            auto const* __restrict tmp_itr1 = itr1 + offset;
            auto const* __restrict tmp_itr2 = itr2 + offset;

            auto* __restrict tmp = itr_x + offset * rows;

            for (size_t i = inner_dim; i < n; ++i)
            {
                auto  a       = tmp_itr1[i];
                auto* tmp_i   = tmp + i * columns;
                auto* tmp_i_1 = tmp + (i - inner_dim) * columns;
                for (size_t j = 0; j < columns; ++j)
                {
                    tmp_i[j] += a * tmp_i_1[j];
                }
            }

            for (int i = end; i >= 0; --i)
            {
                auto  a       = tmp_itr2[i];
                auto* tmp_i   = tmp + i * columns;
                auto* tmp_i_1 = tmp + (i + inner_dim) * columns;

                for (size_t j = 0; j < columns; ++j)
                {
                    tmp_i[j] += a * tmp_i_1[j];
                }
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
}

//-----------------------------------------------------------------------------
void solve_decomposed_aad(
    matrix<double>&       decomposed_aad,
    vector<double>&       x_aad,
    vector<double>&       x,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    const bool            parallelize)
{
    const auto n   = inner_dim * dim;
    const int  end = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;

    auto parallel_eval =
        [&x, &decomposed, &x_aad, &decomposed_aad, inner_dim, n, end](int from, int to)
    {
        auto* __restrict itr_x      = x.data();
        auto const* __restrict itr1 = decomposed[1].data();
        auto const* __restrict itr2 = decomposed[2].data();

        auto* __restrict itr_x_aad = x_aad.data();
        auto* __restrict itr1_aad  = decomposed_aad[1].data();
        auto* __restrict itr2_aad  = decomposed_aad[2].data();

        for (int l = to; l > from; --l)
        {
            const auto offset               = static_cast<size_t>(l - 1) * n;
            auto* __restrict tmp            = itr_x + offset;
            auto const* __restrict tmp_itr1 = itr1 + offset;
            auto const* __restrict tmp_itr2 = itr2 + offset;

            auto* __restrict tmp_aad      = itr_x_aad + offset;
            auto* __restrict tmp_itr1_aad = itr1_aad + offset;
            auto* __restrict tmp_itr2_aad = itr2_aad + offset;

            for (int i = 0; i <= end; ++i)
            {
                //tmp[j] += tmp_itr2[j] * tmp[j + inner_dim];
                tmp_itr2_aad[i] += tmp_aad[i] * tmp[i + inner_dim];
                tmp_aad[i + inner_dim] += tmp_aad[i] * tmp_itr2[i];

                tmp[i] -= tmp_itr2[i] * tmp[i + inner_dim];
            }

            for (size_t i = n - 1; i >= inner_dim; --i)
            {
                //tmp[i] += tmp_itr1[i] * tmp[i - inner_dim];
                tmp_itr1_aad[i] += tmp_aad[i] * tmp[i - inner_dim];
                tmp_aad[i - inner_dim] += tmp_aad[i] * tmp_itr1[i];

                tmp[i] -= tmp_itr1[i] * tmp[i - inner_dim];
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }

    // x = decomposed[0] * x;
    vector<double> tmp_aad(decomposed_aad[0]);
    x /= decomposed[0];
    tmp_aad += x_aad * x;
    x_aad *= decomposed[0];
}

//-----------------------------------------------------------------------------
void solve_decomposed_aad(
    vector<double>&       x_aad,
    const matrix<double>& decomposed,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim)
{
    const auto n   = inner_dim * dim;
    const int  end = static_cast<int>(n) - static_cast<int>(inner_dim) - 1;

    auto const* __restrict itr1 = decomposed[1].data();
    auto const* __restrict itr2 = decomposed[2].data();

    auto* __restrict itr_x_aad = x_aad.data();

    for (int l = static_cast<int>(outer_dim); l > 0; --l)
    {
        const auto offset               = static_cast<size_t>(l - 1) * n;
        auto const* __restrict tmp_itr1 = itr1 + offset;
        auto const* __restrict tmp_itr2 = itr2 + offset;

        auto* __restrict tmp_aad = itr_x_aad + offset;

        for (int i = 0; i <= end; ++i)
        {
            //tmp[i] += tmp_itr2[i] * tmp[i + inner_dim];
            tmp_aad[i + inner_dim] += tmp_aad[i] * tmp_itr2[i];
        }

        for (size_t i = n - 1; i >= inner_dim; --i)
        {
            //tmp[i] += tmp_itr1[i] * tmp[i - inner_dim];
            tmp_aad[i - inner_dim] += tmp_aad[i] * tmp_itr1[i];
        }
    }

    x_aad *= decomposed[0];
}

//-----------------------------------------------------------------------------
void multiply(
    vector<double>&       result,
    const vector<double>& in,
    const matrix<double>& mat,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    double                time_multiplier,
    const bool            parallelize)
{
    const auto n = inner_dim * dim;

    auto parallel_eval =
        [&result, &mat, in, inner_dim, n, outer_dim, time_multiplier](int from, int to)
    {
        const auto offset = static_cast<size_t>(from) * n;

        auto* __restrict itr           = result.data() + offset;
        auto const* __restrict itr_tmp = in.data() + offset;
        auto const* __restrict itr0    = mat[0].data() + offset;
        auto const* __restrict itr1    = mat[1].data() + offset;
        auto const* __restrict itr2    = mat[2].data() + offset;

        for (size_t l = from; l < to; ++l)
        {
            size_t i = 0;
            for (; i < inner_dim; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto b = *itr1;
                const auto c = *itr2;

                *itr += time_multiplier * (b * *itr_tmp + c * *(itr_tmp + inner_dim));
            }

            for (; i < n - inner_dim; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto a = *itr0;
                const auto b = *itr1;
                const auto c = *itr2;

                *itr += time_multiplier * (a * (*(itr_tmp - inner_dim)) + b * (*itr_tmp) +
                                           c * (*(itr_tmp + inner_dim)));
            }

            for (; i < n; ++i, ++itr, ++itr_tmp, ++itr0, ++itr1, ++itr2)
            {
                const auto a = *itr0;
                const auto b = *itr1;

                *itr += time_multiplier * (a * *(itr_tmp - inner_dim) + b * *itr_tmp);
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
};

//-----------------------------------------------------------------------------
void multiply_aad(
    matrix<double>&       mat_aad,
    vector<double>&       in_aad,
    const vector<double>& result_aad,
    const vector<double>& in,
    const matrix<double>& mat,
    const size_t          outer_dim,
    const size_t          dim,
    const size_t          inner_dim,
    double                time_multiplier,
    const bool            parallelize)
{
    const auto n = inner_dim * dim;

    auto parallel_eval =
        [&result_aad, &mat, &mat_aad, &in, &in_aad, inner_dim, n, time_multiplier](int from, int to)
    {
        const auto offset = static_cast<size_t>(to) * n - 1;

        auto const* __restrict itr_aad = result_aad.data() + offset;
        auto const* __restrict itr_tmp = in.data() + offset;

        auto const* __restrict itr0 = mat[0].data() + offset;
        auto const* __restrict itr1 = mat[1].data() + offset;
        auto const* __restrict itr2 = mat[2].data() + offset;

        auto* __restrict itr0_aad    = mat_aad[0].data() + offset;
        auto* __restrict itr1_aad    = mat_aad[1].data() + offset;
        auto* __restrict itr2_aad    = mat_aad[2].data() + offset;
        auto* __restrict itr_tmp_aad = in_aad.data() + offset;

        for (size_t l = to; l > from; --l)
        {
            size_t i = n;

            for (; i > n - inner_dim; --i,
                                      --itr_aad,
                                      --itr_tmp,
                                      --itr0,
                                      --itr1,
                                      --itr2,
                                      --itr_tmp_aad,
                                      --itr0_aad,
                                      --itr1_aad,
                                      --itr2_aad)
            {
                const auto a = *itr0;
                const auto b = *itr1;

                auto tmp_aad = *itr_aad * time_multiplier;
                //*itr += time_multiplier * (a * *(itr_tmp - inner_dim) + b * *itr_tmp);
                *itr0_aad += tmp_aad * *(itr_tmp - inner_dim);
                *itr1_aad += tmp_aad * *(itr_tmp);
                *itr_tmp_aad += tmp_aad * b;
                *(itr_tmp_aad - inner_dim) += tmp_aad * a;
            }

            for (; i > inner_dim; --i,
                                  --itr_aad,
                                  --itr_tmp,
                                  --itr0,
                                  --itr1,
                                  --itr2,
                                  --itr_tmp_aad,
                                  --itr0_aad,
                                  --itr1_aad,
                                  --itr2_aad)
            {
                const auto a = *itr0;
                const auto b = *itr1;
                const auto c = *itr2;

                auto tmp_aad = *itr_aad * time_multiplier;

                //*itr += time_multiplier * (a * (*(itr_tmp - inner_dim)) + b * (*itr_tmp) +c * (*(itr_tmp + inner_dim)));
                *itr0_aad += tmp_aad * *(itr_tmp - inner_dim);
                *itr1_aad += tmp_aad * *(itr_tmp);
                *itr2_aad += tmp_aad * *(itr_tmp + inner_dim);

                *(itr_tmp_aad - inner_dim) += tmp_aad * a;
                *itr_tmp_aad += tmp_aad * b;
                *(itr_tmp_aad + inner_dim) += tmp_aad * c;
            }

            for (; i > 0; --i,
                          --itr_aad,
                          --itr_tmp,
                          --itr0,
                          --itr1,
                          --itr2,
                          --itr_tmp_aad,
                          --itr0_aad,
                          --itr1_aad,
                          --itr2_aad)
            {
                const auto b = *itr1;
                const auto c = *itr2;

                auto tmp_aad = *itr_aad * time_multiplier;
                //*itr += time_multiplier * (b * *itr_tmp + c * *(itr_tmp + inner_dim));
                *itr1_aad += tmp_aad * *(itr_tmp);
                *itr2_aad += tmp_aad * *(itr_tmp + inner_dim);

                *itr_tmp_aad += tmp_aad * b;
                *(itr_tmp_aad + inner_dim) += tmp_aad * c;
            }
        }
    };

    if QUARISMA_UNLIKELY (parallelize && outer_dim > 1)
    {
        quarisma::tools::For(0, static_cast<int>(outer_dim), parallel_eval);
    }
    else
    {
        parallel_eval(0, static_cast<int>(outer_dim));
    }
};
}  // namespace quarisma::tridiagonal_operations
