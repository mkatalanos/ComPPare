#pragma once
#include <benchmark/benchmark.h>

#ifndef COMPPARE_WARMUP_ITERS
#define COMPPARE_WARMUP_ITERS 100
#endif // COMPPARE_WARMUP_ITERS

#ifndef COMPPARE_BENCH_ITERS
#define COMPPARE_BENCH_ITERS 100
#endif // COMPPARE_BENCH_ITERS

namespace ComPPare::param
{
    inline double roi_us = 0.0;

    inline bool inside_gbench = false;

    inline benchmark::State *gb_state = nullptr;
    
    class gbench_scope
    {
    public:
        gbench_scope(benchmark::State &st)
        {
            inside_gbench = true;
            gb_state = &st;
            // gb_state->PauseTiming(); // start paused
        }
        ~gbench_scope()
        {
            inside_gbench = false;
            gb_state = nullptr;
        }
    };

    inline double get_roi_us()
    {
        double last_roi_us = roi_us;
        roi_us = 0.0;
        return last_roi_us;
    }
}
