#pragma once
#include <concepts>
#include <type_traits>
#include <ranges>
#include <string>
#include <ostream>

namespace comppare::internal::concepts
{
    /*
    Concept for valid type that can be streamed to an output stream.
    For example, std::string, int, double, etc; NOT std::vector<int>, std::map<int, int>, etc.
    */
    template <typename T>
    concept Streamable =
        requires(std::ostream &os, T v) { { os << v } -> std::same_as<std::ostream&>; };

    template <typename T>
    concept FloatingPoint = std::floating_point<std::remove_cvref_t<T>>;

    template <typename T>
    concept Integral = std::integral<std::remove_cvref_t<T>>;

    template <typename T>
    concept Arithmetic = FloatingPoint<T> || Integral<T>;

    template <typename T>
    concept String = std::same_as<std::remove_cvref_t<T>, std::string>;

    template <typename T>
    concept Void = std::is_void_v<std::remove_cvref_t<T>>;

    // Concept for a range of arithmetic types, excluding strings
    template <typename R>
    concept RangeOfArithmetic =
        std::ranges::forward_range<std::remove_cvref_t<R>> && Arithmetic<std::remove_cvref_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>> && (!String<std::remove_cvref_t<R>>);

}