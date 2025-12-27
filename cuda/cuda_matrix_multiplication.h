#ifndef __cuda_matrix_multiplication_h__
#define __cuda_matrix_multiplication_h__

#include "MathModule.h"

#ifdef QUARISMA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace quarisma
{

#ifdef QUARISMA_ENABLE_CUDA
/**
 * @brief CUDA kernel wrapper for square matrix multiplication
 * @param d Matrix dimension (d x d matrices)
 * @param ni Number of matrices to multiply
 * @param A_i Input matrices A (device memory)
 * @param B_i Input matrices B (device memory)
 * @param C_i Output matrices C (device memory)
 * @param context CUDA stream for asynchronous execution
 */
MATH_API void cuda_ssqmm(
    int d, int ni, const float* A_i, const float* B_i, float* C_i, const cudaStream_t* context);
#endif

/**
 * @brief CPU interface for CUDA square matrix multiplication
 * @param d Matrix dimension (d x d matrices)
 * @param n Number of matrices to multiply
 * @param x Input matrices A (host memory)
 * @param y Input matrices B (host memory)
 * @param z Output matrices C (host memory)
 */
MATH_API void opcuda_cpu_ssqmm(int d, int n, const float* x, const float* y, float* z);

}  // namespace quarisma
#endif
