#pragma once

#include <vector>
#include <cmath>

struct SaxpyConfig
{
    // saxpy parameters
    size_t N = 1ULL << 10;
    float a = NAN;
    std::vector<float> x, y;

    // benchmark parameters
    double tol = 1e-6;
};

SaxpyConfig init_saxpy(int argc, char **argv);