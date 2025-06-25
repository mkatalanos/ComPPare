#include <string_view>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>
#include <span>

#include "common.hpp"
#include "init_max.hpp"


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
        else if (key == "--iter" && i + 1 < argc)
        {
            cfg.iters = std::stoull(argv[++i]);
        }
        else if (key == "--tol" && i + 1 < argc)
        {
            cfg.tol = std::stod(argv[++i]);
        }
        else if (key == "--warmup")
        {
            cfg.warmup = true;
        }
        else if (key == "--no-warmup")
        {
            cfg.warmup = false;
        }
        else
        {
            throw std::invalid_argument("Unknown argument: " + std::string(key));
        }
    }

    float_t *data;
#ifdef HAVE_CUDA
    cudaMallocHost((void **)&data, cfg.N * sizeof(float_t));
#else
    data = new float_t[cfg.N];
#endif

    // Initialise random number in array
    std::mt19937 rng;
    unsigned int seed = 42;
    rng.seed(seed);

    std::normal_distribution<float_t> dist(-1e5, 1e5);

    for (size_t i = 0; i < cfg.N; ++i)
    {
        data[i] = dist(rng);
    }

    cfg.data = std::span<float_t>(data, cfg.N);

    // Print config
    std::cout << "\n=== Max Reduction Benchmark Parameters ===\n";
    std::cout << "Vector size (N)     : " << cfg.data.size() << '\n';
    std::cout << "Iterations          : " << cfg.iters << '\n';
    std::cout << "Error tolerance     : " << cfg.tol << '\n';
    std::cout << "===================================\n\n";

    return cfg;
}
