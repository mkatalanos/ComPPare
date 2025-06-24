#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace ComPPare
{
    struct ErrorStats
    {
        double max = 0.0;
        double sum = 0.0;
        std::size_t n = 0;

        std::size_t argmax_vec = ~std::size_t(0);
        std::size_t argmax_pos = ~std::size_t(0);

        double mean() const { return n ? sum / n : 0.0; }
    };

    template <std::size_t I = 0, typename... Tup>
    inline std::enable_if_t<I == sizeof...(Tup), void>
    accum_err(const std::tuple<Tup...> &, const std::tuple<Tup...> &,
              ErrorStats &, std::size_t) {}

    template <std::size_t I = 0, typename... Tup>
        inline std::enable_if_t < I<sizeof...(Tup), void>
                                  accum_err(const std::tuple<Tup...> &a,
                                            const std::tuple<Tup...> &b,
                                            ErrorStats &s,
                                            std::size_t vec_idx)
    {
        const auto &va = std::get<I>(a);
        const auto &vb = std::get<I>(b);
        for (std::size_t k = 0; k < va.size(); ++k)
        {
            double e = std::abs(double(va[k] - vb[k]));
            s.sum += e;
            ++s.n;
            if (e > s.max)
            {
                s.max = e;
                s.argmax_vec = vec_idx;
                s.argmax_pos = k;
            }
        }
        accum_err<I + 1>(a, b, s, vec_idx + 1);
    }

    template <typename... Tup>
    ErrorStats error_stats(const std::tuple<Tup...> &a,
                           const std::tuple<Tup...> &b)
    {
        ErrorStats s;
        accum_err(a, b, s, 0);
        return s;
    }

    template <typename... Inputs>
    class InputContext
    {
    public:
        template <typename... Outputs>
        class OutputContext
        {
            using Func = std::function<void(const Inputs &..., Outputs &...,
                                            std::size_t, double &)>;
            using OutTup = std::tuple<Outputs...>;
            struct Impl
            {
                std::string name;
                Func fn;
            };

        public:
            explicit OutputContext(const Inputs &...ins) : inputs_(ins...) {}

            template <typename F>
            void set_reference(std::string name, F &&f)
            {
                impls_.insert(impls_.begin(), {std::move(name), Func(std::forward<F>(f))});
            }

            template <typename F>
            void add(std::string name, F &&f)
            {
                impls_.push_back({std::move(name), Func(std::forward<F>(f))});
            }

            void run(std::size_t iters = 10, double tol = 1e-6, bool warmup = true)
            {
                constexpr int COL_W = 18;
                std::cout << std::left
                          << std::setw(COL_W) << "Name"
                          << std::right
                          << std::setw(COL_W) << "Func µs"
                          << std::setw(COL_W) << "ROI µs"
                          << std::setw(COL_W) << "Ovhd µs";

                constexpr std::size_t NUM_OUT = sizeof...(Outputs);
                for (std::size_t v = 0; v < NUM_OUT; ++v)
                    std::cout << std::setw(COL_W) << ("Max|err|[" + std::to_string(v) + "]")
                              << std::setw(COL_W) << "(MaxErr-idx)"
                              << std::setw(COL_W) << ("Mean|err|[" + std::to_string(v) + "]")
                              << std::setw(COL_W) << ("Total|err|[" + std::to_string(v) + "]");

                std::cout << '\n';

                OutTup ref_out;

                for (std::size_t k = 0; k < impls_.size(); ++k)
                {
                    auto &impl = impls_[k];
                    OutTup outs;
                    double ROI_us = 0.0;

                    if (warmup)
                    {
                        OutTup outs_warmup;
                        double ROI_us_warmup = 0.0;
                        // warmup run of 1 iteration
                        std::apply([&](auto const &...in)
                                   { std::apply(
                                         [&](auto &...out)
                                         { impl.fn(in..., out..., 1, ROI_us_warmup); },
                                         outs_warmup); }, inputs_);
                    }

                    auto t0 = std::chrono::steady_clock::now();
                    /*
                    use std::apply to unpack the inputs and outputs
                    */
                    std::apply([&](auto const &...in)
                               { std::apply(
                                     [&](auto &...out)
                                     { impl.fn(in..., out..., iters, ROI_us); },
                                     outs); }, inputs_);
                    auto t1 = std::chrono::steady_clock::now();

                    double func_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
                    double ovhd_us = func_us - ROI_us;

                    std::array<ErrorStats, NUM_OUT> perVec{};

                    if (k != 0)
                    {
                        std::apply([&](auto &...outR)
                                   { std::apply([&](auto &...outT)
                                                {
                    std::size_t v = 0;
                    ((perVec[v] = error_stats(std::tie(outT), std::tie(outR)), ++v), ...); }, outs); }, ref_out);
                    }

                    // first impl is the reference
                    if (k == 0)
                        ref_out = outs;

                    std::cout << std::left << std::setw(COL_W) << impl.name
                              << std::right << std::setw(COL_W) << std::fixed << std::setprecision(2) << func_us
                              << std::setw(COL_W) << ROI_us
                              << std::setw(COL_W) << ovhd_us;

                    // per-output error
                    for (std::size_t v = 0; v < NUM_OUT; ++v)
                    {
                        const auto &es = perVec[v];
                        double maxerr = (k == 0) ? 0.0 : es.max;
                        double meanerr = (k == 0) ? 0.0 : es.mean();
                        double sumerr = (k == 0) ? 0.0 : es.sum;

                        if (k && maxerr > tol) // fail threshold
                            std::cout
                                << std::setw(COL_W) << std::scientific << maxerr
                                << std::setw(COL_W) << es.argmax_pos; // print max error and location (index)
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
                        if (e.max > tol)
                        {
                            fail = true;
                            break;
                        }

                    if (k && fail)
                        std::cout << "  <-- FAIL";

                    std::cout << '\n';
                } /* for impls */
            } /* run */

        private:
            std::tuple<Inputs...> inputs_;
            std::vector<Impl> impls_;
        }; /* OutputContext */

    }; /* InputContext */

} // namespace compareframework
