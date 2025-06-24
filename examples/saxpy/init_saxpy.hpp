#pragma once

#include <vector>
#include <cmath>

struct SaxpyConfig
{
    // saxpy parameters
    size_t N = 1ULL << 26;
    float a = NAN;
    std::vector<float> x, y;

    // benchmark parameters
    size_t iters = 10;
    double tol = 1e-6;
    bool warmup = true;
};

SaxpyConfig init_saxpy(int argc, char **argv);