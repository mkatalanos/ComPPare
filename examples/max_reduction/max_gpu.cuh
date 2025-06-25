#ifdef HAVE_CUDA
#pragma once
#include <span>
#include "common.hpp"

__global__ void max_kernel(const int thread_max,
                           float_t *__restrict__ max_output,
                           const float_t *__restrict__ d_input);

__global__ void max_kernel_warpsemantics(const int thread_max,
                                         float_t *__restrict__ max_output,
                                         const float_t *__restrict__ d_input);

template <void (*KERNEL)(const int, float_t *__restrict__, const float_t *__restrict__)>
void gpu_max(std::span<const float_t> in,
             float_t &out,
             const size_t iters,
             double &roi_us);

#endif // HAVE_CUDA