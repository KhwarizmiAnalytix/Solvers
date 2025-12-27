#include "common/configure.h"  // IWYU pragma: keep
#include "quarismaTest.h"

#ifdef QUARISMA_ENABLE_CUDA
#include <algorithm>
#include <cstring>

#include "common/pointer.h"
#include "cuda/cuda_matrix_multiplication.h"
#include "matrix_operation/matrix_multiplication.h"
#include "memory/allocator.h"
#include "memory/device.h"
#include "randomgen/mersenne_twister.h"
#include "util/logger.h"
#endif

namespace
{
void test_cuda_matrix_multilpication()
{
#ifdef QUARISMA_ENABLE_CUDA
    using allocator_t = quarisma::allocator<float>;
    auto gen          = quarisma::util::make_ptr_mutable<quarisma::mersenne_twister>(1234567, 0);

    int    d          = 32;
    int    n          = 5;
    int    total_size = d * d * n;
    size_t n_bytes    = total_size * sizeof(float);

    auto* x     = allocator_t::allocate(total_size, quarisma::device_enum::CPU);
    auto* y     = allocator_t::allocate(total_size, quarisma::device_enum::CPU);
    auto* z     = allocator_t::allocate(total_size, quarisma::device_enum::CPU);
    auto* z_cpu = allocator_t::allocate(total_size, quarisma::device_enum::CPU);

    gen->uniforms<float>(x, total_size, 0);
    gen->uniforms<float>(y, total_size, 0);

    memset(z_cpu, 0, n_bytes);
    memset(z, 0, n_bytes);

    quarisma::opcuda_cpu_ssqmm(d, n, x, y, z);

    for (int i = 0; i < n; ++i)
    {
        auto   dim   = static_cast<size_t>(d);
        size_t shift = static_cast<size_t>(i) * dim * dim;
        quarisma::matrix_multiplication(
            false, false, dim, dim, dim, y + shift, dim, x + shift, dim, z_cpu + shift, dim);
    }

    float a = 0;
    for (int i = 0; i < total_size; i++)
    {
        // QUARISMA_LOGF(INFO, "GPU: %f , CPU: %f", z[i], z_cpu[i]);
        a = std::max(a, std::fabs(z[i] - z_cpu[i]));
    }

    QUARISMA_LOGF(INFO, "GPU-CPU error: %f", a);
    EXPECT_LT(a, 0.0000021F);

    allocator_t::free(x, quarisma::device_enum::CPU);
    allocator_t::free(y, quarisma::device_enum::CPU);
    allocator_t::free(z, quarisma::device_enum::CPU);
#endif
}

}  // namespace

QUARISMATEST(Core, GPUMatrixMultiplication)
{
    START_LOG_TO_FILE_NAME(GPUMatrixMultiplication);

    test_cuda_matrix_multilpication();

    END_LOG_TO_FILE_NAME(GPUMatrixMultiplication);

    END_TEST();
}
