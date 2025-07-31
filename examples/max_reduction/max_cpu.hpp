#pragma once

#include <span>

void cpu_max_serial(std::span<const float> in,
                    float &out);

void cpu_max_omp(std::span<const float> in,
                 float &out);

void cpu_max_thread(std::span<const float> in,
                    float &out);