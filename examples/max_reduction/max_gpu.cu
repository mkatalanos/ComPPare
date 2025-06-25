#ifdef HAVE_CUDA
#include "common.hpp"
#include "max_gpu.cuh"
#include <span>

// Minimum size for initialising shared memory in CUDA kernels
__device__ constexpr float_t MINSIZE = std::numeric_limits<float_t>::lowest();

// instantiating the template function for both kernels
template void gpu_max<max_kernel>(std::span<const float_t>, float_t&, const size_t, double&);
template void gpu_max<max_kernel_warpsemantics>(std::span<const float_t>, float_t&, const size_t, double&);

/*
CUDA kernel to find the maximum value in an array
*/
__global__ void max_kernel(const int thread_max,
                                  float_t *__restrict__ max_output,
                                  const float_t *__restrict__ d_input)
{

    __shared__ float_t shared_var[BLOCKSIZE];

    int local_thread = threadIdx.x; // thread number within block

    int global_thread = blockIdx.x * blockDim.x + threadIdx.x; // thread number globally across all blocks

    shared_var[local_thread] = MINSIZE; // Initialize to minimum value

    // Check if the global thread index is within bounds
    if (global_thread >= thread_max) [[unlikely]]
        return;

    // Load data into shared memory
    shared_var[local_thread] = __ldg(&d_input[global_thread]);

    // Ensure all threads have written to shared memory before proceeding
    __syncthreads();

    /*
    Perform reduction in shared memory
    parallel reduction O(log n)

    If there are 32 threads in a block
    only 16 threads will do work in the first iteration,
    to compare the first 16 values with the second 16 values

    then 8 threads will do work in the second iteration,
    so on and so forth until only 1 thread is left
    */
#pragma unroll
    for (int stride = BLOCKSIZE >> 1; stride > 0; stride >>= 1)
    {
        if (local_thread < stride)
        {
            shared_var[local_thread] = float_t_max(shared_var[local_thread], shared_var[local_thread + stride]);
        }
        __syncthreads();
    }

    // Write the result to the output array
    // Only the first thread in the block writes the result
    if (local_thread == 0)
    {
        // Notice I used blockIdx.x here
        // each block writes into its own output location
        max_output[blockIdx.x] = shared_var[0];
    }
}

__global__ void max_kernel_warpsemantics(const int thread_max,
                                                float_t *__restrict__ max_output,
                                                const float_t *__restrict__ d_input)
{
    __shared__ float_t shared_var[32];

    unsigned int local_thread = threadIdx.x;
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_thread < 32)
    {
        shared_var[local_thread] = MINSIZE; // Initialize to minimum value
    }

    if (global_thread >= thread_max) [[unlikely]]
        return;

    float_t val = __ldg(&d_input[global_thread]);

    float_t tmp;

    tmp = __shfl_down_sync(0xffffffff, val, 16);
    val = float_t_max(val, tmp);
    tmp = __shfl_down_sync(0xffffffff, val, 8);
    val = float_t_max(val, tmp);
    tmp = __shfl_down_sync(0xffffffff, val, 4);
    val = float_t_max(val, tmp);
    tmp = __shfl_down_sync(0xffffffff, val, 2);
    val = float_t_max(val, tmp);
    tmp = __shfl_down_sync(0xffffffff, val, 1);
    val = float_t_max(val, tmp);

    int warp_id = local_thread >> 5;
    int lane_id = local_thread & 31;

    if (lane_id == 0)
    {
        shared_var[warp_id] = val;
    }

    __syncthreads();

    if (local_thread < 32) // same as warp_id == 0
    {
        float_t block_max = shared_var[local_thread];
        // Now warp‐shuffle within warp 0 to reduce those num_warps values
        tmp = __shfl_down_sync(0xffffffff, block_max, 16);
        block_max = float_t_max(block_max, tmp);
        tmp = __shfl_down_sync(0xffffffff, block_max, 8);
        block_max = float_t_max(block_max, tmp);
        tmp = __shfl_down_sync(0xffffffff, block_max, 4);
        block_max = float_t_max(block_max, tmp);
        tmp = __shfl_down_sync(0xffffffff, block_max, 2);
        block_max = float_t_max(block_max, tmp);
        tmp = __shfl_down_sync(0xffffffff, block_max, 1);
        block_max = float_t_max(block_max, tmp);

        if (local_thread == 0)
        {
            max_output[blockIdx.x] = block_max;
        }
    }
}

template <void (*KERNEL)(const int, float_t *__restrict__, const float_t *__restrict__)>
void gpu_max(std::span<const float_t> in,
             float_t &out,
             const size_t iters,
             double &roi_us)
{
    float_t *d_input;
    float_t *d_output;

    cudaMalloc((void **)&d_input, in.size() * sizeof(float_t));
    cudaMalloc((void **)&d_output, sizeof(float_t) * ceil((static_cast<float>(in.size()) / static_cast<float>((BLOCKSIZE)))));

    cudaEvent_t cudastart, cudaend;
    cudaEventCreate(&cudastart);
    cudaEventCreate(&cudaend);

    double roi_ms = 0.0;
    for (size_t rep = 0; rep < iters; ++rep)
    {
        cudaMemcpy(d_input, in.data(), in.size() * sizeof(float_t), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float_t) * ceil((static_cast<float>(in.size()) / static_cast<float>((BLOCKSIZE)))));
        int array_size = in.size();
        cudaEventRecord(cudastart);
        while (array_size > BLOCKSIZE)
        {
            int num_blocks = ceil(static_cast<float>(array_size) / static_cast<float>(BLOCKSIZE));
            KERNEL<<<num_blocks, BLOCKSIZE>>>(array_size, d_output, d_input);
            cudaDeviceSynchronize();
            std::swap(d_input, d_output);
            array_size = num_blocks;
        }
        KERNEL<<<1, BLOCKSIZE>>>(array_size, d_output, d_input);
        cudaDeviceSynchronize();
        cudaEventRecord(cudaend);
        cudaEventSynchronize(cudaend);

        float ms = 0;
        cudaEventElapsedTime(&ms, cudastart, cudaend);
        roi_ms += ms;
    }
    roi_us = 1e3 * roi_ms; // µs

    cudaMemcpy(&out, d_output, sizeof(float_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
#endif // HAVE_CUDA