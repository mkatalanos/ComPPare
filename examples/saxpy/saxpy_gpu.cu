#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>
#include "saxpy_gpu.cuh"

__global__ static void saxpy_kernel(const float a,
                                    const float *__restrict__ x,
                                    const float *__restrict__ y_in,
                                    float *__restrict__ y_out,
                                    size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) [[likely]]
    {
        y_out[i] = a * __ldg(&x[i]) * __ldg(&y_in[i]); // Debug this line
    }
}

void gpu_std(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out,
             size_t iters,
             double &roi_us)
{
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUDA error on cudaGetDeviceCount: ") +
            cudaGetErrorString(status));
    }
    if (device_count == 0)
    {
        throw std::runtime_error("No CUDA-capable devices found");
    }

    size_t N = x.size();
    y_out.resize(N);

    float *d_x = nullptr, *d_yin = nullptr, *d_yout = nullptr;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_yin, N * sizeof(float));
    cudaMalloc(&d_yout, N * sizeof(float));
    cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yin, y_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(512), grid((N + block.x - 1) / block.x);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (size_t rep = 0; rep < iters; ++rep)
    {
        saxpy_kernel<<<grid, block>>>(a, d_x, d_yin, d_yout, N);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    roi_us = 1e3 * ms / iters; // µs per iteration

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    cudaMemcpy(y_out.data(), d_yout, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_yin);
    cudaFree(d_yout);
}
