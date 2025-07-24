#pragma once
#include <concepts>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <limits>

namespace comppare::internal::policy
{

    /* 
    Concept for a valid Error Policy

    It requires:
    - metric_count() to return the number of metrics
    - metric_name(std::size_t i) to return the name of the metric at index i
    - compute_error(const Val &a, const Val &b, double tol) to compute the error
      between two values a and b with a given tolerance tol -- or not
    - metric(std::size_t) to return the value of the metric
    - is_fail() to return true if the error exceeds the tolerance
    */
    template <typename Val, typename EP>
    concept ErrorPolicy = requires 
    // static members
    {
        { EP::metric_count() } -> std::convertible_to<std::size_t>;
        { EP::metric_name(std::size_t{}) } -> std::convertible_to<std::string_view>;
    } && 
    // compute error -- either with or without tolerance
    (
        requires(EP ep, const Val &a, const Val &b, double t) { 
            ep.compute_error(a, b, t); 
        } || 
        requires(EP ep, const Val &a, const Val &b) { 
            ep.compute_error(a, b); 
        }
    ) &&
    // metric() returns value of the metric
        requires(EP ep, std::size_t i) {
             { ep.metric(i) } -> std::convertible_to<double>; 
        } && 
    // is_fail() -- either with or without tolerance
    (
        requires(EP ep, double t) {
            { ep.is_fail(t) } -> std::convertible_to<bool>; 
        } || 
        requires(EP ep) { 
            { ep.is_fail() } -> std::convertible_to<bool>; 
        }
    );

    template <class EP, class V, class Tol>
    inline void compute_error(EP &ep, const V &a, const V &b, Tol tol)
    {
        if constexpr (requires { ep.compute_error(a, b, tol); })
            ep.compute_error(a, b, tol);
        else
            ep.compute_error(a, b);
    }

    template <class EP, class Tol>
    inline bool is_fail(const EP &ep, Tol tol)
    {
        if constexpr (requires { ep.is_fail(tol); })
            return ep.is_fail(tol);
        else
            return ep.is_fail();
    }

    namespace autopolicy
    {
        template <typename T>
        concept Arithmetic = std::integral<std::remove_cvref_t<T>> || std::floating_point<std::remove_cvref_t<T>>;

        template <typename T>
        concept String = std::same_as<std::remove_cvref_t<T>, std::string>;

        // Concept for a range of arithmetic types, excluding strings
        template <typename R>
        concept RangeOfArithmetic =
            std::ranges::forward_range<std::remove_cvref_t<R>> && Arithmetic<std::remove_cvref_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>> && (!String<std::remove_cvref_t<R>>);

        template <typename T>
        concept SupportedByAutoPolicy =
            Arithmetic<T> || String<T> || RangeOfArithmetic<T>;

        /* 
        Error Policy for scalar/numbers
        */
        template <typename T>
            requires Arithmetic<T>
        class ArithmeticErrorPolicy
        {
            T error_ = T(0);
            static constexpr std::array names{"Total|err|"};

        public:
            static constexpr std::size_t metric_count() { return 1; }
            static constexpr std::string_view metric_name(std::size_t) { return names[0]; }

            T metric(std::size_t) const { return error_; }
            bool is_fail(T tol = std::numeric_limits<T>::epsilon()) const { return error_ > tol; }

            void compute_error(const T &a, const T &b,
                               T tol = std::numeric_limits<T>::epsilon())
            {
                if constexpr (!std::is_floating_point_v<T>)
                {
                    if (!std::isfinite(a) || !std::isfinite(b))
                    {
                        error_ = std::numeric_limits<T>::quiet_NaN();
                        return;
                    }
                }

                T e = std::abs(a - b);
                if (e <= tol)
                    return;
                error_ = e;
            }
        };

        /*
        Error Policy for strings
        */
        class StringEqualPolicy
        {
            bool neq_{false};

        public:
            static constexpr std::size_t metric_count() { return 1; }
            static constexpr std::string_view metric_name(std::size_t) { return "neq"; }

            double metric(std::size_t) const { return neq_; }
            bool is_fail() const { return neq_; }

            void compute_error(const std::string &a, const std::string &b) { neq_ = !(a == b); }
        };

        /*
        Error Policy for ranges of arithmetic types
        eg. std::vector<int>, std::deque<float>, etc.
        */
        template <typename R>
            requires RangeOfArithmetic<R>
        class RangeErrorPolicy
        {
            using T = std::remove_cvref_t<std::ranges::range_value_t<R>>;

            T max_error_ = T(0);
            T total_error_ = T(0);
            std::size_t elem_cnt_ = 0;

            static constexpr std::array names{"Max|err|", "Mean|err|", "Total|err|"};

        public:
            // ——— static interface ————————————————————————————
            static constexpr std::size_t metric_count() { return names.size(); }
            static constexpr std::string_view metric_name(std::size_t i) { return names[i]; }

            T get_mean() const
            {
                return elem_cnt_ ? total_error_ / static_cast<T>(elem_cnt_) : T(0);
            }

            // ——— accessors ————————————————————————————————————————
            T metric(std::size_t i) const
            {
                switch (i)
                {
                case 0:
                    return max_error_;
                case 1:
                    return get_mean();
                case 2:
                    return total_error_;
                default:
                    return T(0);
                }
            }

            bool is_fail(T tol = std::numeric_limits<T>::epsilon()) const
            {
                if constexpr (!std::is_floating_point_v<T>)
                {
                    if (!std::isfinite(max_error_) || !std::isfinite(total_error_))
                        return true; // NaN or Inf
                }
                return max_error_ > tol;
            }

            // ——— comparison ————————————————————————————————————————
            void compute_error(const R &a, const R &b,
                               T tol = std::numeric_limits<T>::epsilon())
            {
                if constexpr (!std::is_floating_point_v<T>)
                {
                    if (!std::isfinite(a) || !std::isfinite(b))
                    {
                        max_error_ = std::numeric_limits<T>::quiet_NaN();
                        total_error_ = std::numeric_limits<T>::quiet_NaN();
                        elem_cnt_ = 0;
                        return;
                    }
                }
                auto ia = std::ranges::begin(a);
                auto ib = std::ranges::begin(b);
                for (; ia != std::ranges::end(a) && ib != std::ranges::end(b); ++ia, ++ib)
                {
                    T diff = std::abs(*ia - *ib);
                    if (diff <= tol)
                        continue;
                    total_error_ += diff;
                    max_error_ = std::max(max_error_, diff);
                    ++elem_cnt_;
                }
            }
        };

        /*
        AutoPolicy is a helper to deduce the appropriate error policy
        Currently only supports: scalar types, strings, and ranges of arithmetic types (see above concepts)
        */
        template <typename T>
        struct AutoPolicy;

        template <typename T>
            requires Arithmetic<T>
        struct AutoPolicy<T>
        {
            using type = ArithmeticErrorPolicy<std::remove_cvref_t<T>>;
        };
        template <typename T>
            requires String<T>
        struct AutoPolicy<T>
        {
            using type = StringEqualPolicy;
        };

        template <typename T>
            requires RangeOfArithmetic<T>
        struct AutoPolicy<T>
        {
            using type = RangeErrorPolicy<T>;
        };

        template <typename T>
        using AutoPolicy_t = typename AutoPolicy<T>::type;
    }
}
