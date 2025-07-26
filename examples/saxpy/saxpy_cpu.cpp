#include <vector>
#include <thread>

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

void cpu_par(float a,
             const std::vector<float> &x,
             const std::vector<float> &y_in,
             std::vector<float> &y_out)
{
    size_t N = x.size();
    y_out.resize(N);

    unsigned nThreads = std::thread::hardware_concurrency();
    size_t chunk = (N + nThreads - 1) / nThreads;

    auto worker = [&](size_t start, size_t end)
    {
        for (size_t i = start; i < end && i < N; ++i)
        {
            y_out[i] = a * x[i] + y_in[i];
        }
    };

    HOTLOOPSTART;
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (unsigned t = 0; t < nThreads; ++t)
    {
        size_t start = t * chunk;
        size_t end = start + chunk;
        threads.emplace_back(worker, start, end);
    }
    for (auto &th : threads)
        th.join();
    HOTLOOPEND;
}