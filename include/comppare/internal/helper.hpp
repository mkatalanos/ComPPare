#pragma once
#include <string>
#include <string_view>
#include <stdexcept>
#include <cstring>

#include <comppare/internal/config.hpp>

namespace comppare::internal::helper
{
    static inline void parse_args(int argc, char **argv)
    {
        auto getValue = [&](std::string_view token, const char *nextArg)
        {
            auto eq = token.find('=');
            std::string_view valstr;
            if (eq != std::string_view::npos)
            {
                valstr = token.substr(eq + 1);
            }
            else if (nextArg)
            {
                valstr = nextArg;
            }
            else
            {
                throw std::invalid_argument(
                    std::string(token) + " requires a value");
            }
            size_t idx = 0;
            int v = std::stoi(std::string(valstr), &idx);
            if (idx != valstr.size())
                throw std::invalid_argument(
                    std::string("invalid integer for ") + std::string(token));
            return v;
        };

        for (int i = 1; i < argc; ++i)
        {
            std::string_view arg = argv[i];
            if (arg.rfind("--warmup", 0) == 0)
            {
                const char *next = (arg.find('=') == std::string_view::npos && i + 1 < argc)
                                       ? argv[++i]
                                       : nullptr;
                int w = getValue(arg, next);
                comppare::config::set_warmup_iters(w);
            }
            else if (arg.rfind("--iter", 0) == 0)
            {
                const char *next = (arg.find('=') == std::string_view::npos && i + 1 < argc)
                                       ? argv[++i]
                                       : nullptr;
                int n = getValue(arg, next);
                comppare::config::set_bench_iters(n);
            }
        }
    }
} // namespace comppare::internal::helper