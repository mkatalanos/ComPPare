#pragma once
// #ifdef HAVE_GOOGLE_BENCHMARK
#include <utility>
#include <tuple>
#include <ostream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <benchmark/benchmark.h>

#include "comppare/plugin/plugin.hpp"

namespace comppare::plugin::google_benchmark
{
    class state
    {
    public:
        state(const state &) = delete;
        state &operator=(const state &) = delete;
        state(state &&) = delete;
        state &operator=(state &&) = delete;

        static state &instance()
        {
            static state inst;
            return inst;
        }

        static void set_state(benchmark::State &st)
        {
            instance().st_ = &st;
        }

        static benchmark::State &get_state()
        {
            if (!instance().st_)
                throw std::runtime_error("Benchmark state is not set.");
            return *instance().st_;
        }

    private:
        state() = default;
        benchmark::State *st_ = nullptr;
    };

    // TODO: Merge this into GoogleBenchmarkPlugin
    class google_benchmark_manager
    {
    public:
        google_benchmark_manager() = default;
        ~google_benchmark_manager() = default;

        void initialize(int &argc, char **argv)
        {
            parse_args(argc, argv);
            print_benchmark_header();

            int n = static_cast<int>(bench_argv_.size());
            benchmark::Initialize(&n, bench_argv_.data());
            benchmark::ReportUnrecognizedArguments(n, bench_argv_.data());
        }

        template <typename Func, typename... Args>
        benchmark::internal::Benchmark *add_gbench(const char *name, Func f, Args &&...args)
        {
            std::tuple<Args...> cargs(std::forward<Args>(args)...);

            auto benchptr = benchmark::RegisterBenchmark(
                name,
                [f, cargs = std::move(cargs)](benchmark::State &st) mutable
                {
                    comppare::plugin::google_benchmark::state::set_state(st);
                    std::apply([&](auto &&...unpacked)
                               { f(std::forward<decltype(unpacked)>(unpacked)...); }, cargs);
                    benchmark::ClobberMemory();
                });

            return benchptr;
        }

        void run()
        {
            benchmark::RunSpecifiedBenchmarks();
            benchmark::Shutdown();
        }

    private:
        std::vector<char *> bench_argv_;

        static std::string strip_quotes(const std::string &s)
        {
            if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
                return s.substr(1, s.size() - 2);
            return s;
        }

        void parse_args(int &argc, char **argv)
        {
            std::vector<std::string> bench_flags;
            int write_i = 1;

            for (int read_i = 1; read_i < argc; ++read_i)
            {
                std::string cur(argv[read_i]);

                if (cur.rfind("--gbench=", 0) == 0)
                {
                    // --gbench="...v..."
                    std::string v = cur.substr(strlen("--gbench="));
                    bench_flags.push_back(strip_quotes(v));
                }
                else if (cur == "--gbench" && read_i + 1 < argc)
                {
                    // --gbench "v"
                    std::string v = argv[++read_i];
                    bench_flags.push_back(strip_quotes(v));
                }
                else
                {
                    // keep this in argv[]
                    argv[write_i++] = argv[read_i];
                }
            }
            argc = write_i; // new argc, gbench flags removed

            bench_argv_.clear();
            bench_argv_.push_back(strdup(argv[0]));
            for (auto &f : bench_flags)
                bench_argv_.push_back(strdup(f.c_str()));
        }

        void print_benchmark_header()
        {
            std::cout << "\n"
                      << "============================================================\n"
                      << std::setw(2) << std::right << ">>>"
                      << " Google Benchmark\n"
                      << std::setw(2) << std::right << ">>>"
                      << " Args:\n";

            for (int i = 0; i < (int)bench_argv_.size(); ++i)
            {
                std::cout << std::setw(2) << std::right << " "
                          << "  [" << i << "] " << std::quoted(bench_argv_[i]) << "\n";
            }

            std::cout << "============================================================\n";
        }
    };

    template <class InTup, class OutTup>
    class GoogleBenchmarkPlugin final : public Plugin<InTup, OutTup>
    {
        using Self = GoogleBenchmarkPlugin<InTup, OutTup>;
        using Func = std::function<void(const InTup &, OutTup &)>;
        comppare::plugin::google_benchmark::google_benchmark_manager gb_;

    public:
        GoogleBenchmarkPlugin(const GoogleBenchmarkPlugin &) = delete;
        GoogleBenchmarkPlugin &operator=(const GoogleBenchmarkPlugin &) = delete;

        static std::shared_ptr<Self> instance()
        {
            static std::shared_ptr<Self> inst{new Self};
            return inst;
        }

        template <class F>
        benchmark::internal::Benchmark *register_impl(const std::string &name,
                                                      F &&user_fn,
                                                      const InTup &inputs,
                                                      OutTup &outs)
        {
            return std::apply([&](auto const &...in_vals)
                              { return std::apply([&](auto &&...outs_vals)
                                                  { return gb_.add_gbench(name.c_str(),
                                                                          std::forward<F>(user_fn),
                                                                          in_vals..., outs_vals...); }, outs); }, inputs);
        }

        void initialize(int &argc, char **argv) override
        {
            gb_.initialize(argc, argv);
        }
        void run() override
        {
            gb_.run();
        }

    private:
        GoogleBenchmarkPlugin() = default;
    };

#define PLUGIN_HOTLOOPEND                                                          \
    benchmark::State &st = comppare::plugin::google_benchmark::state::get_state(); \
    for (auto _ : st)                                                              \
    {                                                                              \
        hotloop_body();                                                            \
    }

}

// #endif // HAVE_GOOGLE_BENCHMARK