#pragma once

#include <vector>

void gpu_std(float a,
    const std::vector<float> &x,
    const std::vector<float> &y_in,
    std::vector<float> &y_out,
    size_t iters,
    double &roi_us);
