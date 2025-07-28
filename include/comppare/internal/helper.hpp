#pragma once
#include <charconv>
#include <string>
#include <string_view>
#include <stdexcept>
#include <cstdint>

#include <comppare/internal/config.hpp>

namespace comppare::internal::helper
{
    template <typename T>
    T get_arg_value(std::string_view option, const char *nextArg)
    {
        std::string_view valstr;
        if (auto eq = option.find('='); eq != std::string_view::npos)
            valstr = option.substr(eq + 1);
        else if (nextArg)
            valstr = nextArg;
        else
            throw std::invalid_argument(std::string(option) + " requires a value");

        
        if constexpr (std::same_as<T, std::string>)
        {
            return std::string(valstr);
        }
        else if constexpr (std::is_integral_v<T>) // if integral type
        {
            size_t idx = 0;
            std::string s{valstr};

            if constexpr (std::is_signed_v<T>) // if signed
            {
                long long tmp = std::stoll(s, &idx);
                if (idx != s.size()) // if not all characters were processed
                    throw std::invalid_argument("invalid integer for " + std::string(option));

                if (tmp < std::numeric_limits<T>::min() ||
                    tmp > std::numeric_limits<T>::max()) // if out of range for T
                    throw std::out_of_range("integer out of range for " + std::string(option));

                return static_cast<T>(tmp);
            }
            else // unsigned
            {
                if (!s.empty() && s.front() == '-') // if negative -- reject
                    throw std::invalid_argument("invalid unsigned integer for " + std::string(option));

                unsigned long long tmp = std::stoull(s, &idx); 
                if (idx != s.size()) // if not all characters were processed
                    throw std::invalid_argument("invalid unsigned integer for " + std::string(option));

                if (tmp > std::numeric_limits<T>::max()) // if out of range for T
                    throw std::out_of_range("unsigned integer out of range for " + std::string(option));

                return static_cast<T>(tmp);
            }
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            size_t idx = 0;
            long double tmp = std::stold(std::string(valstr), &idx);
            if (idx != valstr.size())
                throw std::invalid_argument("invalid floating-point for " + std::string(option));
            return static_cast<T>(tmp);
        }
        else
        {
            static_assert(std::is_arithmetic_v<T> || std::same_as<T, std::string>,
                          "get_arg_value supports only arithmetic types or std::string");
        }
    }

    static inline void parse_args(int argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string_view arg = argv[i];

            auto get_next_arg_if_needed = [&](std::string_view a) -> const char *
            {
                return (a.find('=') == std::string_view::npos && i + 1 < argc)
                           ? argv[++i]
                           : nullptr;
            };

            if (arg.rfind("--warmup", 0) == 0)
            {
                auto w = get_arg_value<std::uint64_t>(arg, get_next_arg_if_needed(arg));
                comppare::config::set_warmup_iters(w);
            }
            else if (arg.rfind("--iter", 0) == 0)
            {
                auto n = get_arg_value<std::uint64_t>(arg, get_next_arg_if_needed(arg));
                comppare::config::set_bench_iters(n);
            }
            else if (arg.rfind("--tolerance", 0) == 0)
            {
                auto tol = get_arg_value<long double>(arg, get_next_arg_if_needed(arg));
                comppare::config::set_all_fp_tolerance(tol);
            }
        }
    }
} // namespace comppare::internal::helper