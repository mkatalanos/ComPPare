#include <chrono>
#include <algorithm>
#include <limits>
#include <thread>
#include <vector>
#include <omp.h>
#include <cmath>
#include "max_cpu.hpp"
#include <iostream>

void cpu_max_serial(std::span<const float_t> in,
                    float_t &out,
                    const size_t iters,
                    double &roi_us)
{
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float_t result;
    for (size_t rep = 0; rep < iters; ++rep)
    {
        result = *std::max_element(in.data(), in.data() + in.size());
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();

    out = result;
    roi_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
}

void cpu_max_omp(std::span<const float_t> in,
                 float_t &out,
                 const size_t iters,
                 double &roi_us)
{
    auto cpu_start = std::chrono::high_resolution_clock::now();

    float_t result = std::numeric_limits<float_t>::lowest();
    for (size_t rep = 0; rep < iters; ++rep)
    {
#pragma omp parallel for reduction(max : result) schedule(static)
        for (size_t i = 0; i < in.size(); ++i)
            result = std::max(result, in[i]);
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    out = result;
    roi_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
}

void cpu_max_thread(std::span<const float_t> in,
                    float_t &out,
                    const size_t iters,
                    double &roi_us)
{
    const unsigned num_threads = std::thread::hardware_concurrency();

    std::vector<float_t> local_max(num_threads);
    std::vector<std::thread> threads(num_threads);
    std::vector<std::pair<size_t, size_t> > ranges(num_threads);

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

    float_t result;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t rep = 0; rep < iters; ++rep)
    {
        // launch workers
        for (unsigned t = 0; t < num_threads; ++t)
        {
            threads[t] = std::thread([&, t]()
                                     { local_max[t] = *std::max_element(in.data() + ranges[t].first, in.data() + ranges[t].second); });
        }
        // join and reduce
        for (auto &th : threads)
            th.join();
        result = *std::max_element(local_max.begin(), local_max.end());
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    out = result;
    roi_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
}