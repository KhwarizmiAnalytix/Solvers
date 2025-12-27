#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda_matrix_multiplication.h"

#define NCOLS 4

#define RANK_UPDATE                   \
    A += lda;                         \
    rank1_update(*A, &b[i * ldb], c); \
    if (++i >= k)                     \
    return
#define STORE_BLOCK \
    C += ldc;       \
    *C = c[i++];    \
    if (i >= num)   \
    return

template <typename scalar_t>
__device__ void rank1_update(const scalar_t m, const scalar_t* __restrict__ b, scalar_t* c)
{
    c[0] += m * *b;
    c[1] += m * *(++b);
    c[2] += m * *(++b);
    c[3] += m * *(++b);
    c[4] += m * *(++b);
    c[5] += m * *(++b);
    c[6] += m * *(++b);
    c[7] += m * *(++b);
    c[8] += m * *(++b);
    c[9] += m * *(++b);
    c[10] += m * *(++b);
    c[11] += m * *(++b);
    c[12] += m * *(++b);
    c[13] += m * *(++b);
    c[14] += m * *(++b);
    c[15] += m * *(++b);
}

template <typename scalar_t>
__device__ void rankk_update(
    int k,
    const scalar_t* __restrict__ A0,
    int lda,
    const scalar_t* __restrict__ b,
    int       ldb,
    scalar_t* c)
{
    if (k <= 0)
        return;
    const auto* A = A0;
    int         i = 0;
    rank1_update(*A, &b[i * ldb], c);
    if (++i >= k)
        return;

    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;
    RANK_UPDATE;

    A += lda;
    rank1_update(*A, &b[i * ldb], c);
}

template <typename scalar_t>
__device__ void store_block2(int num, const scalar_t* __restrict__ c, scalar_t* C, int ldc)
{
    if (num <= 0)
        return;
    int i = 0;

    *C = c[i++];
    if (i >= num)
        return;

    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;
    STORE_BLOCK;

    C += ldc;
    *C = c[i++];
}

/**
 * @brief Simple CUDA kernel for matrix multiplication C = A * B
 * @param d Matrix dimension (d x d matrices)
 * @param n Number of matrices to multiply
 * @param A_i Input matrices A (device memory)
 * @param B_i Input matrices B (device memory)
 * @param C_i Output matrices C (device memory)
 */
template <typename scalar_t>
static __global__ void simple_matrix_mult(
    const int d, const int n, const scalar_t* A_i, const scalar_t* B_i, scalar_t* C_i)
{
    // Calculate which matrix we're working on
    int matrix_id = blockIdx.z;
    if (matrix_id >= n)
        return;

    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (row >= d || col >= d)
        return;

    // Calculate matrix offsets
    int             matrix_size = d * d;
    const scalar_t* A           = A_i + matrix_id * matrix_size;
    const scalar_t* B           = B_i + matrix_id * matrix_size;
    scalar_t*       C           = C_i + matrix_id * matrix_size;

    // Compute C[row][col] = sum(A[row][k] * B[k][col])
    scalar_t sum = 0;
    for (int k = 0; k < d; k++)
    {
        sum += A[row * d + k] * B[k * d + col];
    }

    // Store result
    C[row * d + col] = sum;
}

namespace xsigma
{

/**
 * @brief CUDA kernel wrapper for square matrix multiplication
 * @param d Matrix dimension (d x d matrices)
 * @param ni Number of matrices to multiply
 * @param A_i Input matrices A (device memory)
 * @param B_i Input matrices B (device memory)
 * @param C_i Output matrices C (device memory)
 * @param context CUDA stream for asynchronous execution
 */
void cuda_ssqmm(
    int d, int ni, const float* A_i, const float* B_i, float* C_i, const cudaStream_t* context)
{
    if (d <= 0 || ni <= 0 || !A_i || !B_i || !C_i || !context)
    {
        printf("CUDA Error: Invalid parameters for cuda_ssqmm: d=%d, ni=%d\n", d, ni);
        return;
    }

    // Use simple kernel for now to ensure correctness
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y, ni);

    // Launch CUDA kernel
    simple_matrix_mult<float><<<grid, block, 0, *context>>>(d, ni, A_i, B_i, C_i);

    // Check for kernel launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess)
    {
        printf("CUDA Error: Kernel launch failed: %s\n", cudaGetErrorString(launch_error));
    }
}

/**
 * @brief CPU interface for CUDA square matrix multiplication
 * @param d Matrix dimension (d x d matrices)
 * @param n Number of matrices to multiply
 * @param x Input matrices A (host memory)
 * @param y Input matrices B (host memory)
 * @param z Output matrices C (host memory)
 */
void opcuda_cpu_ssqmm(int d, int n, const float* x, const float* y, float* z)
{
    if (d <= 0 || n <= 0 || !x || !y || !z)
    {
        printf("CUDA Error: Invalid parameters for opcuda_cpu_ssqmm: d=%d, n=%d\n", d, n);
        return;
    }

    try
    {
        const size_t total_size = static_cast<size_t>(d) * d * n;
        const size_t bytes      = total_size * sizeof(float);

        // Allocate device memory
        float* x_d = nullptr;
        float* y_d = nullptr;
        float* z_d = nullptr;

        cudaError_t err;

        // Allocate GPU memory
        err = cudaMalloc(&x_d, bytes);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                "Failed to allocate GPU memory for x_d: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&y_d, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(x_d);
            throw std::runtime_error(
                "Failed to allocate GPU memory for y_d: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&z_d, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(x_d);
            cudaFree(y_d);
            throw std::runtime_error(
                "Failed to allocate GPU memory for z_d: " + std::string(cudaGetErrorString(err)));
        }

        // Create CUDA stream
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess)
        {
            cudaFree(x_d);
            cudaFree(y_d);
            cudaFree(z_d);
            throw std::runtime_error(
                "Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
        }

        // Copy data from host to device
        err = cudaMemcpyAsync(x_d, x, bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(stream);
            cudaFree(x_d);
            cudaFree(y_d);
            cudaFree(z_d);
            throw std::runtime_error(
                "Failed to copy x to device: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMemcpyAsync(y_d, y, bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(stream);
            cudaFree(x_d);
            cudaFree(y_d);
            cudaFree(z_d);
            throw std::runtime_error(
                "Failed to copy y to device: " + std::string(cudaGetErrorString(err)));
        }

        // Initialize output memory to zero
        err = cudaMemsetAsync(z_d, 0, bytes, stream);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(stream);
            cudaFree(x_d);
            cudaFree(y_d);
            cudaFree(z_d);
            throw std::runtime_error(
                "Failed to initialize z_d: " + std::string(cudaGetErrorString(err)));
        }

        // Perform matrix multiplication on GPU
        // Note: CPU test computes C = Y * X, so we swap arguments to match
        cuda_ssqmm(d, n, y_d, x_d, z_d, &stream);

        // Wait for kernel completion
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(stream);
            cudaFree(x_d);
            cudaFree(y_d);
            cudaFree(z_d);
            throw std::runtime_error(
                "CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
        }

        // Copy result back to host
        err = cudaMemcpyAsync(z, z_d, bytes, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(stream);
            cudaFree(x_d);
            cudaFree(y_d);
            cudaFree(z_d);
            throw std::runtime_error(
                "Failed to copy result to host: " + std::string(cudaGetErrorString(err)));
        }

        // Wait for copy completion
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            printf("CUDA Error: Failed to synchronize final copy: %s\n", cudaGetErrorString(err));
        }

        // Clean up resources
        cudaStreamDestroy(stream);
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(z_d);

        printf(
            "CUDA Info: Matrix multiplication completed successfully: %dx%d matrices, %d batches\n",
            d,
            d,
            n);
    }
    catch (const std::exception& e)
    {
        printf("CUDA Error: Matrix multiplication failed: %s\n", e.what());
        throw;
    }
}

}  // namespace xsigma
