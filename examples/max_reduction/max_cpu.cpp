#include <chrono>
#include <algorithm>
#include <limits>
#include <thread>
#include <vector>
#include <omp.h>
#include <cmath>

#include <comppare/comppare.hpp>

#include "max_cpu.hpp"

void cpu_max_serial(std::span<const float> in, float &out)
{
    HOTLOOPSTART;
    out = *std::max_element(in.data(), in.data() + in.size());
    HOTLOOPEND;
}

void cpu_max_omp(std::span<const float> in, float &out)
{
    HOTLOOPSTART;
#pragma omp parallel for reduction(max : out) schedule(static)
        for (size_t i = 0; i < in.size(); ++i)
            out = std::max(out, in[i]);
    HOTLOOPEND;
}

void cpu_max_thread(std::span<const float> in, float &out)
{
    const unsigned num_threads = std::thread::hardware_concurrency();

    std::vector<float> local_max(num_threads);
    std::vector<std::thread> threads(num_threads);
    std::vector<std::pair<size_t, size_t>> ranges(num_threads);

    const size_t N = in.size();
    const size_t base = std::floor(static_cast<float>(N) / static_cast<float>(num_threads));
    const size_t remainder = N % num_threads;

    size_t start = 0;
    for (unsigned t = 0; t < num_threads; ++t)
    {
        size_t end = start + base + (t < remainder ? 1 : 0);
        ranges[t].first = start;
        ranges[t].second = end;
        start = end;
    }

    HOTLOOPSTART;
    // launch workers
    for (unsigned t = 0; t < num_threads; ++t)
    {
        threads[t] = std::thread([&, t]()
                                 { local_max[t] = *std::max_element(in.data() + ranges[t].first, in.data() + ranges[t].second); });
    }
    // join and reduce
    for (auto &th : threads)
        th.join();
    out = *std::max_element(local_max.begin(), local_max.end());
    
    HOTLOOPEND;
}