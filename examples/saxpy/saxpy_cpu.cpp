#include <numeric>
#include <vector>
#include <thread>
#include <chrono>
#if (HAVE_NVHPC)
#include <algorithm>
#include <execution>
#include <cuda_runtime.h>
#endif
#include "saxpy_cpu.hpp"

void cpu_std(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out,
             size_t iters,
             double &roi_us)
{
    size_t N = x.size();
    y_out.resize(N);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t rep = 0; rep < iters; ++rep)
    {
        for (size_t i = 0; i < N; ++i)
        {
            y_out[i] = a * x[i] + y_in[i];
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    roi_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

void cpu_par(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out,
             size_t iters,
             double &roi_us)
{
    size_t N = x.size();
    y_out.resize(N);

    unsigned nThreads = std::thread::hardware_concurrency();
    size_t chunk = (N + nThreads - 1) / nThreads;

    auto worker = [&](size_t start, size_t end)
    {
        for (size_t i = start; i < end && i < N; ++i)
        {
            y_out[i] = a * x[i] + y_in[i];
        }
    };

    auto t0 = std::chrono::steady_clock::now();
    for (size_t rep = 0; rep < iters; ++rep)
    {
        std::vector<std::thread> threads;
        threads.reserve(nThreads);
        for (unsigned t = 0; t < nThreads; ++t)
        {
            size_t start = t * chunk;
            size_t end = start + chunk;
            threads.emplace_back(worker, start, end);
        }
        for (auto &th : threads)
            th.join();
    }
    auto t1 = std::chrono::steady_clock::now();

    roi_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

#if (HAVE_NVHPC)
void gpu_std_par(float a,
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

    cudaFree(nullptr);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (size_t rep = 0; rep < iters; ++rep)
    {
        std::transform(
            std::execution::par_unseq,
            x.begin(), x.end(),
            y_in.begin(),
            y_out.begin(),
            [a](float xi, float yi)
            { return a * xi + yi; });
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    roi_us = 1e3 * ms / iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif