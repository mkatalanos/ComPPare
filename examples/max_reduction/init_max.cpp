#include <string_view>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>
#include <span>

#include "init_max.hpp"
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif


MaxConfig init_max(int argc, char **argv)
{
    MaxConfig cfg;

    // Parse flags
    for (int i = 1; i < argc; ++i)
    {
        std::string_view key = argv[i];

        if (key == "--size" && i + 1 < argc)
        {
            cfg.N = 1ULL << std::stoull(argv[++i]);
        }
    }

    float *data;
#ifdef HAVE_CUDA
    cudaMallocHost((void **)&data, cfg.N * sizeof(float));
#else
    data = new float[cfg.N];
#endif

    // Initialise random number in array
    std::mt19937 rng;
    unsigned int seed = 42;
    rng.seed(seed);

    std::normal_distribution<float> dist(-1e5, 1e5);

    for (size_t i = 0; i < cfg.N; ++i)
    {
        data[i] = dist(rng);
    }

    cfg.data = std::span<float>(data, cfg.N);

    // Print config
    std::cout << "\n=== Max Reduction Benchmark Parameters ===\n";
    std::cout << "Vector size (N)     : " << cfg.data.size() << '\n';
    std::cout << "===================================\n\n";

    return cfg;
}
