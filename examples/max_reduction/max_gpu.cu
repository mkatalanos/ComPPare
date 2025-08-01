#ifdef HAVE_CUDA

#include "max_gpu.cuh"
#include <span>
#include <comppare/comppare.hpp>

// Minimum size for initialising shared memory in CUDA kernels
__device__ constexpr float MINSIZE = std::numeric_limits<float>::lowest();

// instantiating the template function for both kernels
template void gpu_max<max_kernel>(std::span<const float>, float &);
template void gpu_max<max_kernel_warpsemantics>(std::span<const float>, float &);

// CUDA kernel to find the maximum value in an array
__global__ void max_kernel(const int thread_max,
                           float *__restrict__ max_output,
                           const float *__restrict__ d_input)
{

    __shared__ float shared_var[BLOCKSIZE];

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
            shared_var[local_thread] = fmaxf(shared_var[local_thread], shared_var[local_thread + stride]);
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

/*
warp shuffle reduction
NVIDIA BLOG: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
CUDA Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=warp#warp-shuffle-functions

__shfl_down_sync is part of the "synchronized data exchange" warp primitives
it allows threads within in a warp to exchange data even if they are in registers

__shfl_down_sync(0xffffffff, warp_max, 16);
moves data from thread i to i-16 (or named as "lane" in correct terminology)
then compares the value in thread i with the value in thread i-16
and returns the maximum of the two values
(0xffffffff is the mask for all threads in the warp, meaning all threads can participate in the shuffle)

this is repeated until all threads in the warp have been compared (16, 8, 4, 2, 1)

**
NOTE: __reduce_max_sync() can be used instead of the manual reduction for int32 only
in our case we are using fp32, so we have to do it manually
**

*/
#define warp_size 32
static __forceinline__ __device__ void warp_reduce_max(float &warpmax)
{
#pragma unroll 
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
    {
        warpmax = fmaxf(warpmax, __shfl_down_sync(0xffffffff, warpmax, offset));
    }
}

// CUDA kernel to find the maximum value in an array using warp shuffle semantics
__global__ void max_kernel_warpsemantics(const int thread_max,
                                         float *__restrict__ max_output,
                                         const float *__restrict__ d_input)
{

    unsigned int local_thread = threadIdx.x;
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory only for second level of reduction
    __shared__ float shared_var[32];

    // Initialize shared memory for the second level of reduction
    if (local_thread < 32)
    {
        shared_var[local_thread] = MINSIZE; // Initialize to minimum value
    }

    // Check if the global thread index is within bounds
    if (global_thread >= thread_max) [[unlikely]]
        return;

    // load data from global memory into registers
    float warp_max = __ldg(&d_input[global_thread]);

    warp_reduce_max(warp_max);

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
    if (local_thread < warp_size) // same as warp_id == 0
    {
        // load the warp_max value from shared memory
        float block_max = shared_var[local_thread];

        warp_reduce_max(block_max);

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
template <void (*KERNEL)(const int, float *__restrict__, const float *__restrict__)>
void gpu_max(std::span<const float> in,
             float &out)
{
    // Create Device Pointer
    float *d_input;
    float *d_output;

    cudaMalloc((void **)&d_input, in.size() * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float) * ceil((static_cast<float>(in.size()) / static_cast<float>((BLOCKSIZE)))));

    GPU_HOTLOOPSTART;
    // copy input data to device memory and initialize output memory
    cudaMemcpy(d_input, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float) * ceil((static_cast<float>(in.size()) / static_cast<float>((BLOCKSIZE)))));

    int array_size = in.size();

    GPU_MANUAL_TIMER_START;
    /*
    Each kernel reduces the input array per block and writes the maximum value into the output array.
    The output array is then reduced again in the next iteration until only one value remains.
    */
    while (array_size > 1)
    {
        // Calculate the number of blocks needed for the current array size
        int num_blocks = ceil(static_cast<float>(array_size) / static_cast<float>(BLOCKSIZE));
        KERNEL<<<num_blocks, BLOCKSIZE>>>(array_size, d_output, d_input);
        cudaDeviceSynchronize();
        // Swap input and output pointers for the next iteration
        std::swap(d_input, d_output);
        array_size = num_blocks;
    }
    // Final swap to ensure the final result is in d_output
    std::swap(d_input, d_output);
    GPU_MANUAL_TIMER_END;
    GPU_HOTLOOPEND;

    cudaMemcpy(&out, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
}
#endif // HAVE_CUDA