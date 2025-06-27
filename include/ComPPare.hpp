#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "ErrorStats.hpp"

namespace ComPPare
{
    /*
    InputContext class template to hold input parameters for the comparison framework.
    */
    template <typename... Inputs>
    class InputContext
    {
    public:
        /*
        OutputContext class template to hold output parameters and functions for the comparison framework.
        */
        template <typename... Outputs>
        class OutputContext
        {
        private:
            // Alias for the user-provided function signature:
            // (const Inputs&..., Outputs&..., size_t iterations, double& roi_us)
            using Func = std::function<void(const Inputs &..., Outputs &...,
                                            size_t, double &)>;

            // Alias for the error statistics class
            using ErrorStats = ComPPare::internal::ErrorStats;

            // Holds each input and output type in a tuple
            using InTup = std::tuple<Inputs...>;
            using OutTup = std::tuple<Outputs...>;

            // Tuple to hold all input parameters/data
            InTup inputs_;
            // Reference output tuple to hold the outputs of the first implementation
            OutTup ref_out;

            // Struct to hold the function name and function pointer
            struct Impl
            {
                std::string name;
                Func fn;
            };
            // Vector to hold all implementations
            std::vector<Impl> impls_;

        public:
            // Constructor to initialize the OutputContext with inputs
            // This is used to hold and pass the same input arguments/data for all implementations
            explicit OutputContext(const Inputs &...ins) : inputs_(ins...) {}

            // Function to set a reference implementation
            template <typename F>
            void set_reference(std::string name, F &&f)
            {
                impls_.insert(impls_.begin(), {std::move(name), Func(std::forward<F>(f))});
            }

            // Function to add an implementation to the comparison
            template <typename F>
            void add(std::string name, F &&f)
            {
                impls_.push_back({std::move(name), Func(std::forward<F>(f))});
            }

            /*
            Runs the comparison for all added implementations.
            Arguments:
            iterations -- the number of iteratins inside each custom function
            tol -- the tolerance for error comparison
            warmup -- if true, runs a warmup iteration before the actual timing
            */
            void run(size_t iters = 10, double tol = 1e-6, bool warmup = true)
            {
                // Number of output arguments -- sizeof... is used to get the number of elements in a pack
                // https://en.cppreference.com/w/cpp/language/sizeof....html
                constexpr size_t NUM_OUT = sizeof...(Outputs);

                // Print header for the output table
                constexpr int COL_W = 18;
                std::cout << std::left
                          << std::setw(COL_W) << "Name"
                          << std::right
                          << std::setw(COL_W) << "Func µs"
                          << std::setw(COL_W) << "ROI µs"
                          << std::setw(COL_W) << "Ovhd µs";

                for (size_t v = 0; v < NUM_OUT; ++v)
                    std::cout << std::setw(COL_W) << ("Max|err|[" + std::to_string(v) + "]")
                              << std::setw(COL_W) << "(MaxErr-idx)"
                              << std::setw(COL_W) << ("Mean|err|[" + std::to_string(v) + "]")
                              << std::setw(COL_W) << ("Total|err|[" + std::to_string(v) + "]");

                std::cout << std::endl;

                // Main loop to iterate over all implementations
                for (size_t k = 0; k < impls_.size(); ++k)
                {
                    // Get the current implementation
                    auto &impl = impls_[k];

                    OutTup outs;         // output tuple to hold the outputs of the current implementation
                    double roi_us = 0.0; // roi time in microseconds

                    // warmup run of 1 iteration
                    if (warmup)
                    {
                        OutTup outs_warmup;
                        double roi_us_warmup = 0.0;
                        std::apply([&](auto const &...in)
                                   { std::apply(
                                         [&](auto &...out)
                                         { impl.fn(in..., out..., 1, roi_us_warmup); },
                                         outs_warmup); }, inputs_);
                    }

                    // Measure the time taken by the function
                    auto t0 = std::chrono::steady_clock::now();
                    /*
                    use std::apply to unpack the inputs and outputs completely to do 1 function call of the implementation
                    this is equivalent to calling:
                    impl.fn(inputs[0], inputs[1], ..., outputs[0], outputs[1], iters, roi_us);
                    */
                    std::apply([&](auto const &...in)
                               { std::apply(
                                     [&](auto &...out)
                                     { impl.fn(in..., out..., iters, roi_us); },
                                     outs); }, inputs_);
                    auto t1 = std::chrono::steady_clock::now();
                    // end of timing

                    // Calculate the time taken by the function in microseconds
                    double func_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    double ovhd_us = func_us - roi_us;

                    // Create an array to hold error statistics for each output
                    std::array<ErrorStats, NUM_OUT> perVec{};

                    // calculate error if not reference implementation
                    if (k != 0)
                    {
                        /*
                        unpack the reference outputs and current outputs one by one
                        it is like calling:
                        for (size_t v = 0; v < NUM_OUT; ++v)
                        {
                            perVec[v] = error_stats(ref_out[v], outs[v], tol);
                        }
                        */
                        std::apply([&](auto &...outR)
                                   { std::apply([&](auto &...outT)
                                                {
                    size_t v = 0;
                    ((perVec[v].error_stats(outT, outR, tol), ++v), ...); }, outs); }, ref_out);
                    }

                    // first impl is the reference
                    if (k == 0)
                        ref_out = outs;

                    // Print the results for the current implementation
                    std::cout << std::left << std::setw(COL_W) << impl.name
                              << std::right << std::setw(COL_W) << std::fixed << std::setprecision(2) << func_us
                              << std::setw(COL_W) << roi_us
                              << std::setw(COL_W) << ovhd_us;

                    // per-output print error
                    for (size_t v = 0; v < NUM_OUT; ++v)
                    {
                        const auto &es = perVec[v];
                        double maxerr = (k == 0) ? 0.0 : es.max();
                        double meanerr = (k == 0) ? 0.0 : es.mean();
                        double sumerr = (k == 0) ? 0.0 : es.sum();

                        if (k && maxerr > tol) // fail threshold
                            std::cout
                                << std::setw(COL_W) << std::scientific << maxerr
                                << std::setw(COL_W) << es.max_pos(); // print max error and location (index)
                        else
                            std::cout
                                << std::setw(COL_W) << std::scientific << 0.0
                                << std::setw(COL_W) << "—";

                        // print mean and total
                        std::cout
                            << std::setw(COL_W) << std::scientific << meanerr
                            << std::setw(COL_W) << std::scientific << sumerr;
                    }
                    bool fail = false;
                    for (const auto &e : perVec)
                        if (e.max() > tol)
                        {
                            fail = true;
                            break;
                        }

                    if (k && fail)
                        std::cout << "  <-- FAIL";

                    std::cout << '\n';
                } /* for impls */
            } /* run */
        }; /* OutputContext */
    }; /* InputContext */
} // namespace ComPPare
