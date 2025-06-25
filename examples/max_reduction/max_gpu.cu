#ifdef HAVE_CUDA
#include "common.hpp"
#include "max_gpu.cuh"
#include <span>

// Minimum size for initialising shared memory in CUDA kernels
__device__ constexpr float_t MINSIZE = std::numeric_limits<float_t>::lowest();

// instantiating the template function for both kernels
template void gpu_max<max_kernel>(std::span<const float_t>, float_t &, const size_t, double &);
template void gpu_max<max_kernel_warpsemantics>(std::span<const float_t>, float_t &, const size_t, double &);

// CUDA kernel to find the maximum value in an array
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
        // each block writes into its own output location
        max_output[blockIdx.x] = shared_var[0];
    }
}

// CUDA kernel to find the maximum value in an array using warp shuffle semantics
__global__ void max_kernel_warpsemantics(const int thread_max,
                                         float_t *__restrict__ max_output,
                                         const float_t *__restrict__ d_input)
{

    unsigned int local_thread = threadIdx.x;
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory only for second level of reduction
    __shared__ float_t shared_var[32];

    // Initialize shared memory for the second level of reduction
    if (local_thread < 32)
    {
        shared_var[local_thread] = MINSIZE; // Initialize to minimum value
    }

    // Check if the global thread index is within bounds
    if (global_thread >= thread_max) [[unlikely]]
        return;

    // load data from global memory into registers
    float_t warp_max = __ldg(&d_input[global_thread]);

    float_t tmp;

    /*
    warp shuffle reduction
    NVIDIA BLOG: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    CUDA Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=warp#warp-shuffle-functions

    __shfl_down_sync is part of the "synchronized data exchange" warp primitives
    it allows threads within in a warp to exchange data even if they are in registers

    __shfl_down_sync(0xffffffff, warp_max, 16);
    moves data from thread i to i-16 (or named as "lane" in warp terminology)
    then compares the value in thread i with the value in thread i-16
    and returns the maximum of the two values
    (0xffffffff is the mask for all threads in the warp, meaning all threads can participate in the shuffle)

    this is repeated until all threads in the warp have been compared

    **
    NOTE: __reduce_max_sync() can be used instead of the manual reduction for int32 only
    in our case we are using fp32 or fp64, so we have to do it manually
    **

    */
    tmp = __shfl_down_sync(0xffffffff, warp_max, 16);
    warp_max = float_t_max(warp_max, tmp);
    tmp = __shfl_down_sync(0xffffffff, warp_max, 8);
    warp_max = float_t_max(warp_max, tmp);
    tmp = __shfl_down_sync(0xffffffff, warp_max, 4);
    warp_max = float_t_max(warp_max, tmp);
    tmp = __shfl_down_sync(0xffffffff, warp_max, 2);
    warp_max = float_t_max(warp_max, tmp);
    tmp = __shfl_down_sync(0xffffffff, warp_max, 1);
    warp_max = float_t_max(warp_max, tmp);

    int warp_id = local_thread >> 5;
    int lane_id = local_thread & 31;

    // Store the maximum value from each warp in shared memory
    if (lane_id == 0)
    {
        shared_var[warp_id] = warp_max;
    }
    __syncthreads();

    /*
    after first stage of warp shuffle reduction,
    maximum of 32 warp_max values is stored in shared memory (1024 max threads / 32 threads per warp = 32 max warps)

    now we need to reduce those (at most) 32 values to a single maximum value
    using another warp shuffle reduction
    */
    if (local_thread < 32) // same as warp_id == 0
    {
        // load the warp_max value from shared memory
        float_t block_max = shared_var[local_thread];

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

/*
Template function to perform the maximum reduction on the GPU
KERNEL is a template parameter that can be either max_kernel or max_kernel_warpsemantics

**
NOTE:
This function only times the kernel execution time,
it does not include the time for memory allocation and data transfer.
**

*/
template <void (*KERNEL)(const int, float_t *__restrict__, const float_t *__restrict__)>
void gpu_max(std::span<const float_t> in,
             float_t &out,
             const size_t iters,
             double &roi_us)
{
    // Create Device Pointer
    float_t *d_input;
    float_t *d_output;

    cudaMalloc((void **)&d_input, in.size() * sizeof(float_t));
    cudaMalloc((void **)&d_output, sizeof(float_t) * ceil((static_cast<float>(in.size()) / static_cast<float>((BLOCKSIZE)))));
    // Create CUDA events for timing
    cudaEvent_t cudastart, cudaend;
    cudaEventCreate(&cudastart);
    cudaEventCreate(&cudaend);

    double roi_ms = 0.0;
    for (size_t rep = 0; rep < iters; ++rep)
    {
        // copy input data to device memory and initialize output memory
        cudaMemcpy(d_input, in.data(), in.size() * sizeof(float_t), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float_t) * ceil((static_cast<float>(in.size()) / static_cast<float>((BLOCKSIZE)))));

        int array_size = in.size();
        cudaEventRecord(cudastart);

        /*
        Each kernel reduces the input array per block and writes the maximum value into the output array.
        The output array is then reduced again in the next iteration until only one value remains.
        */
        while (array_size > BLOCKSIZE)
        {
            // Calculate the number of blocks needed for the current array size
            int num_blocks = ceil(static_cast<float>(array_size) / static_cast<float>(BLOCKSIZE));
            KERNEL<<<num_blocks, BLOCKSIZE>>>(array_size, d_output, d_input);
            cudaDeviceSynchronize();
            // Swap input and output pointers for the next iteration
            std::swap(d_input, d_output);
            array_size = num_blocks;
        }
        // Final reduction for the last block
        KERNEL<<<1, BLOCKSIZE>>>(array_size, d_output, d_input);
        cudaDeviceSynchronize();
        cudaEventRecord(cudaend);
        cudaEventSynchronize(cudaend);

        float ms = 0;
        cudaEventElapsedTime(&ms, cudastart, cudaend);
        // accumulate the time for each iteration
        roi_ms += ms;
    }
    roi_us = 1e3 * roi_ms; // µs

    cudaMemcpy(&out, d_output, sizeof(float_t), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(cudastart);
    cudaEventDestroy(cudaend);
}
#endif // HAVE_CUDA