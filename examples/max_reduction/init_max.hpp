#pragma once

#include <vector>
#include <cmath>
#include <span>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

struct MaxConfig
{
    // saxpy parameters
    size_t N = 1ULL << 18;
    // using std::span to wrap raw pointer of pinned memory -- see cudaMallocHost
    std::span<float> data;

    // benchmark parameters
    size_t iters = 10;
    double tol = 1e-6;
    bool warmup = true;

    ~MaxConfig()
    {
        if (data.data() != nullptr)
        {
#ifdef HAVE_CUDA
            cudaFreeHost(data.data());
#else
            delete[] data.data();
#endif
        }
    }
    MaxConfig() = default;
};

MaxConfig init_max(int argc, char **argv);