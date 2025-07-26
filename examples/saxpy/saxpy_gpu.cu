#include <vector>
#include <stdexcept>
#include <string>

#include <comppare/comppare.hpp>

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
        y_out[i] = a + __ldg(&x[i]) + __ldg(&y_in[i]); // Debug this line
    }
}

void gpu_std(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out)
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

    float *d_x = nullptr, *d_y_in = nullptr, *d_y_out = nullptr;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y_in, N * sizeof(float));
    cudaMalloc(&d_y_out, N * sizeof(float));
    cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_in, y_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(512), grid((N + block.x - 1) / block.x);

    GPU_HOTLOOPSTART;
    saxpy_kernel<<<grid, block>>>(a, d_x, d_y_in, d_y_out, N);
    GPU_HOTLOOPEND;

    cudaMemcpy(y_out.data(), d_y_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y_in);
    cudaFree(d_y_out);
}
