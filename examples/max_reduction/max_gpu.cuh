#ifdef HAVE_CUDA
#pragma once
#include <span>
#include "common.hpp"

__global__ void max_kernel(const int thread_max,
                           float *__restrict__ max_output,
                           const float *__restrict__ d_input);

__global__ void max_kernel_warpsemantics(const int thread_max,
                                         float *__restrict__ max_output,
                                         const float *__restrict__ d_input);

template <void (*KERNEL)(const int, float *__restrict__, const float *__restrict__)>
void gpu_max(std::span<const float> in,
             float &out);

#endif // HAVE_CUDA