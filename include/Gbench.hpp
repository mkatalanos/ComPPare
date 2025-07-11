#pragma once
#include <benchmark/benchmark.h>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

namespace ComPPare::internal
{

    class gbench_manager
    {
    public:
        gbench_manager() = default;
        ~gbench_manager() = default;

        void initialize(int &argc, char **argv)
        {
            parse_args(argc, argv);
            int n = static_cast<int>(bench_argv_.size());
            ::benchmark::Initialize(&n, bench_argv_.data());
            ::benchmark::ReportUnrecognizedArguments(n, bench_argv_.data());
        }

        template <typename Func, typename... Args>
        inline void add_gbench(const char *name, Func f, Args &&...args)
        {
            using TupleT = std::tuple<Args...>;
            TupleT cargs(std::forward<Args>(args)...);
            ::benchmark::RegisterBenchmark(
                name,
                [f, cargs = std::move(cargs)](::benchmark::State &st) mutable
                {
                    for (auto _ : st)
                    {
                        std::apply([&](auto &&...unpacked)
                                   { f(std::forward<decltype(unpacked)>(unpacked)...); }, cargs);
                        ::benchmark::ClobberMemory();
                    }
                });
        }

        void run()
        {
            ::benchmark::RunSpecifiedBenchmarks();
            ::benchmark::Shutdown();
        }

    private:
        std::vector<char *> bench_argv_;

        void parse_args(int &argc, char **argv)
        {
            // collect indices to remove
            std::vector<int> to_remove;
            for (int i = 1; i < argc; ++i)
            {
                std::string cur(argv[i]);
                if (cur.rfind("--gbench=", 0) == 0)
                {
                    collect_flags(cur.substr(9));
                    to_remove.push_back(i);
                }
                else if (cur == "--gbench" && i + 1 < argc)
                {
                    collect_flags(argv[++i]);
                    to_remove.push_back(i - 1);
                    to_remove.push_back(i);
                }
            }
            bench_argv_.push_back(strdup(argv[0]));
            std::sort(to_remove.begin(), to_remove.end(), std::greater<>());
            for (int idx : to_remove)
            {
                for (int j = idx; j + 1 < argc; ++j)
                    argv[j] = argv[j + 1];
                --argc;
            }
        }

        void collect_flags(const std::string &payload)
        {
            std::istringstream ss(payload);
            std::string token;
            while (ss >> token)
            {
                bench_argv_.push_back(strdup(token.c_str()));
            }
        }
    };
}
