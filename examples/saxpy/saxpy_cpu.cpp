#include <vector>
#include <algorithm>
#ifndef __clang__
#include <execution>
#endif
#include <omp.h>

#include <comppare/comppare.hpp>

#include "saxpy_cpu.hpp"

void cpu_std(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out)
{
    size_t N = x.size();
    y_out.resize(N);

    HOTLOOPSTART;
    for (size_t i = 0; i < N; ++i)
    {
        y_out[i] = a * x[i] + y_in[i];
    }
    HOTLOOPEND;
}

void cpu_omp(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out)
{
    size_t N = x.size();
    y_out.resize(N);

    HOTLOOPSTART;
#pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
        y_out[i] = a * x[i] + y_in[i];
    }
    HOTLOOPEND;
}

#ifndef __clang__
void cpu_par(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out)
{
    size_t N = x.size();
    y_out.resize(N);

    HOTLOOPSTART;
    std::transform(
        std::execution::par_unseq,
        x.begin(), x.end(),
        y_in.begin(),
        y_out.begin(),
        [a](auto xi, auto yi)
        {
            return a * xi + yi;
        });
    HOTLOOPEND;
}
#endif