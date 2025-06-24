#pragma once

#include <vector>

void cpu_std(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out,
             std::size_t iters,
             double &roi_us);

void cpu_par(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out,
             size_t iters,
             double &roi_us);

#if (HAVE_NVHPC)
void gpu_std_par(float a,
                 const std::vector<float> &x,
                 const std::vector<float> &y_in,
                 std::vector<float> &y_out,
                 size_t iters,
                 double &roi_us);
#endif