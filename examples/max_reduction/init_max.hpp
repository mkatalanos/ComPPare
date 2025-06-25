#pragma once

#include <vector>
#include <cmath>
#include <span>
#include "common.hpp"

struct MaxConfig
{
    // saxpy parameters
    size_t N = 1ULL << 26;
    // using std::span to wrap raw pointer of pinned memory -- see cudaMallocHost
    std::span<float_t> data;

    // benchmark parameters
    size_t iters = 10;
    double tol = 1e-6;
    bool warmup = true;
};

MaxConfig init_max(int argc, char **argv);