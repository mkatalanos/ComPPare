#pragma once

#include "common.hpp"
#include <span>

void cpu_max_serial(std::span<const float_t> in,
                    float_t &out,
                    const size_t iters,
                    double &roi_us);

void cpu_max_omp(std::span<const float_t> in,
                 float_t &out,
                 const size_t iters,
                 double &roi_us);

void cpu_max_thread(std::span<const float_t> in,
                    float_t &out,
                    const size_t iters,
                    double &roi_us);